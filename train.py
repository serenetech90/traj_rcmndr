import seaborn as sb
import argParser as parser
import time
import sys
import psutil
import logging
# from models import g2k_lstm_mcr as mcr
# from models import gsk_lstm_cell as cell
from models import g2k_lstm_mc as MC
from matplotlib.pyplot import imread
import networkx_graph as nx_g
import load_traj as load
import tensorflow as tf
import helper
import numpy as np
import relational_inf_models.nri_learned as nri
from matplotlib import pyplot as py
# import tensorflow.python.util.deprecation as deprecation
import os
import glob
# import skimage.measure as sk

# reduce tf messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# deprecation._PRINT_DEPRECATION_WARNINGS = False

# Public vars
true_path = []
euc_loss = []
fde = []
num_nodes = 0
target_traj = []
pred_path = []
e = 0
frame = 1
num_targets = 0
num_end_targets = 0
attn = []

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.warning, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    args = parser.ArgsParser()
    train(args.parser.parse_args())
    return

# TODO use glstm to predict path along with neighborhood boundaries using inner estimated soft attention mechanism.
# take relative distance + orthogonality between people vislets (biased by visual span width)
# extract outputs from glstm as follows: neighborhood influence (make message passing between related pedestrians)
# then transform using mlp the new hidden states mixture into (x,y) euclidean locations.


def train(args):
    tf_graph = tf.Graph()
    out_graph = tf.Graph()
    # tf.enable_eager_execution()
    with tf.Session(graph=out_graph).as_default() as out_sess:
        # tf.enable_eager_execution()
        # with out_sess.as_default():
        # choice = 2 #int(input("Select 1. For training, 2. For Validation \n"))
        # if choice == 1:
        for l in {args.leaveDataset}:
            true_path = []
            num_nodes = 0
            target_traj = []
            pred_path = []
            e = 0
            frame = 1
            num_targets = 0
            num_end_targets = 0
            attn = []

            datasets = {1}

            for d in datasets:
                dataloader = load.DataLoader(args=args, datasets=[0, 1, 2, 3, 4, 5], start=d, sel=0)
                parent_dir = dataloader.parent_dir
                time_log = open(os.path.join(parent_dir, dataloader.used_data_dirs[d], 'training_Tlog.txt'), 'w')
                log_count = parent_dir + '/log/stanford/hyang/aol_osc_counts_{0}.txt'.format(
                    d)
                log_count_f = open(log_count, 'w')
                log_dir = open(parent_dir + '/log/stanford/hyang/aol_osc_ade_log_kfold_{0}.csv'.format(
                    d), 'w')
                log_dir_fde = open(parent_dir + '/log/stanford/hyang/aol_osc_fde_log_kfold_{0}.csv'.format(
                    d), 'w')
                save_dir = parent_dir+'/save/stanford/hyang/'

                # traj = dataloader.load_trajectories(data_file=dataloader.sel_file)

                # TODO implement k-fold cross validation + check why pred_path is all zeros (bug in GridLSTMCell)
                graph = nx_g.online_graph(args)

                print(dataloader.sel_file)
                dataloader.reset_data_pointer()
                flag = True
                pid = psutil.Process(os.getpid())
                # checkpoint_path = os.path.join('/home/serene/PycharmProjects/multimodaltraj_2/ablations/save')
                # /home/serene/PycharmProjects/multimodaltraj_2/log
                # pred = tf.Variable()
                # target = tf.Variable()

                # Squared-Errors
                with tf.Session(graph=out_graph) as sess:
                    while e < args.num_epochs:
                        e_start = time.time()
                        batch, target_traj, _ = dataloader.next_step()

                        if len(batch) == 0:
                            if d == max(datasets):
                                dataloader.dataset_pointer = min(datasets)
                            dataloader.reset_data_pointer()
                            continue

                        print('session started at Epoch = ', e)

                        if e == 0 and flag:
                            nri_obj = nri.nri_learned(args=args, sess=sess)
                            dim = int(args.neighborhood_size / args.grid_size)
                            # only init hidden states at 1st epoch, 1st batch, no resetting is happening after that
                            hidden_state = tf.random_normal(shape=(args.num_freq_blocks, args.rnn_size)) #
                            hidden_filters = tf.truncated_normal(shape=[8, 1, 1], dtype=tf.float32)
                            
                            ctxt_img_path = glob.glob(dataloader.current_dir + 'hyang_{}.jpg'.format(dataloader.sel))
                            ctxt_img = tf.convert_to_tensor(imread(ctxt_img_path[0]), dtype=tf.float32)

                            ctxt_img_pd = tf.convert_to_tensor(
                                tf.pad(ctxt_img, paddings=tf.constant([[1, 1, ], [0, 1], [0, 0]])),
                                dtype=tf.float32)
                            width = int(ctxt_img_pd.shape.dims[0])
                            height = int(ctxt_img_pd.shape.dims[1])
    
                            ctxt_img_pd = tf.expand_dims(ctxt_img_pd, axis=0)
                            _2dconv = tf.nn.conv2d(input=ctxt_img_pd,
                                                   filter=tf.random_normal(
                                                       shape=[width - dim + 1, height - dim + 1, 3, 1],
                                                       dtype=tf.float32),
                                                   padding='VALID', strides=[1, 1, 1, 1])
    
                            _2dconv = tf.squeeze(_2dconv).eval()
                            _2dconv = args.lambda_param * _2dconv
                            hidden_state = hidden_state.eval()
                            flag = False

                            init_w = tf.initializers.random_normal(mean=0, stddev=1, seed=0,
                                                                   dtype=tf.float32)

                            nghood_enc = helper.neighborhood_stat_vis_loc_encoder(
                                hidden_size=args.rnn_size,
                                hidden_len=dim,
                                num_layers=args.num_layers,
                                grid_size=args.grid_size,
                                embedding_size=args.embedding_size,
                                dim=dim,
                                dropout=args.dropout)

                            with tf.variable_scope('nghood_init', reuse=True):
                                out_init = tf.zeros(dtype=tf.float32, shape=(
                                    dim + 2, dim))  # (args.grid_size * (args.grid_size / 2))))
                                c_hidden_init = tf.zeros(dtype=tf.float32, shape=(
                                    dim + 2, dim))  # (args.grid_size * (args.grid_size / 2))))
    
                            stat_mask = tf.zeros(shape=(dim + 2, args.obs_len), dtype=tf.float32)
                            stat_mask += tf.expand_dims(
                                tf.range(start=0, limit=1, delta=(1 / args.obs_len), dtype=tf.float32), axis=0)
                            # static_mask_nd = stat_mask.eval()

                            _2dconv_in = tf.matmul(stat_mask, _2dconv)
                            # _2dconv_valin = _2dconv_in.eval()

                        # with tf_graph.as_default():
                        for b in range(dataloader.num_batches):
                            print('Batch # {} Started'.format(b))
                            graph_t = graph.ConstructGraph(current_batch=batch, framenum=frame,
                                                           future_traj=target_traj)

                            # TODO: validate whether networkx class restores the node values correctly.
                            batch_v = list(graph_t.get_node_attr(param='node_pos_list').values())
                            if len(np.array(batch_v).shape) > 2:
                                if frame > 0:
                                    batch_v = np.array(batch_v)[:, frame - 1:frame + args.obs_len]
                                else:
                                    batch_v = np.array(batch_v)[:, frame:frame + args.obs_len]
                                # batch_v = np.array(batch_v)[:,frame-1:frame + args.obs_len]
                                rl_num_nodes = batch_v.shape[0]
                                batch_v = np.linalg.norm(batch_v, axis=2).squeeze()
                            else:
                                # batch_v = np.array(batch_v)[frame:frame + args.obs_len]
                                dataloader.reset_data_pointer()
                                break

                            batch_v = np.transpose(batch_v)
                            
                            try:
                                num_nodes = args.maxNumPeds # batch_v.shape[1]
                            except IndexError:
                                dataloader.reset_data_pointer()
                                break

                            nri_obj.target_traj0_ten = tf.placeholder(dtype=tf.float32)
                            pred_path = tf.placeholder(dtype=tf.float32)
                            
                            nri_obj.target_traj0_ten = sess.run(
                                [nri_obj.target_traj0_ten],
                                {
                                    nri_obj.target_traj0_ten: nri.extract_ten_dict(target_traj, np.zeros((num_nodes, args.pred_len,
                                                                                   args.input_size)))}
                            )[0]
                            
                            pred_path = sess.run([pred_path], {pred_path: tf.random_normal(shape=(10, 2, 12, num_nodes),
                                                               dtype=tf.float32).eval()})[0]
                            
                            krnl_mdl = MC.g2k_lstm_mc(in_features=nghood_enc.input,  # MC.g2k_lstm_mc
                                                        num_nodes=pred_path.shape[3],
                                                        obs_len=args.obs_len,
                                                        lambda_reg=args.lambda_param,
                                                        sess_g=out_graph)
    
                            with tf.variable_scope('weight_input',reuse=True):
                                
                                weight_i = tf.Variable(name='weight_i',
                                                           initial_value=init_w(
                                                               shape=(rl_num_nodes, args.num_freq_blocks)),
                                                       trainable=True, dtype=tf.float32)
                                weight_ii = tf.Variable(name='weight_ii',
                                                        initial_value=init_w(
                                                            shape=(dim+2, args.obs_len)),
                                                        trainable=True, dtype=tf.float32)
                            
                            tf.initialize_variables(var_list=[weight_i, weight_ii]).run()
    
                            inputs = tf.convert_to_tensor(batch_v, dtype=tf.float32)
                            inputs = tf.matmul(inputs, weight_i)
                            inputs = tf.matmul(weight_ii, inputs)
    
                            # Embed features set into fixed-shaped compact tensor [8x8]
                            tf.initialize_variables(
                                var_list=[krnl_mdl.weight_v, krnl_mdl.bias_v]).run()  # , krnl_mdl.embed_vis
                            tf.initialize_variables(var_list=[krnl_mdl.weight_c,
                                                              krnl_mdl.weight_o, krnl_mdl.cost, nri_obj.l2norm_vec]).run() # krnl_mdl.attn,
                            
                            # sess.run(fetches=tf.initialize_all_variables())

                            start_t = time.time()
                            # adj_mat = tf.zeros(shape=(num_nodes, num_nodes))
                            # if b > 0 and b % 20 == 0:
                            #     sess.graph.clear_collection(name='variables')
                            for framenum in batch:
                                # try:
                                true_path.append(batch[framenum])
                                # compute relative locations and relative vislets
                                st_embeddings, new_hidden_state, ng_output, c_hidden_state = \
                                    sess.run([nghood_enc.input, nghood_enc.state_f00_b00_c,
                                              nghood_enc.output, nghood_enc.c_hidden_state],
                                             feed_dict={nghood_enc.input: inputs.eval(),
                                                        nghood_enc.state_f00_b00_c: hidden_state,
                                                        nghood_enc.output: out_init.eval(),
                                                        nghood_enc.stat_input: _2dconv_in.eval(),
                                                        nghood_enc.c_hidden_state: c_hidden_init.eval()})
    
                                tr_hidden_state = tf.reshape(hidden_state, shape=(16, 80, 1))
                                _conv1d = tf.nn.conv1d(input=tr_hidden_state,
                                                       filters=hidden_filters,
                                                       padding='VALID', stride=[1, 8, 1])
    
                                # tr_hidden_state = _conv1d(tr_hidden_state)
                                # tr_hidden_state = tr_hidden_state.eval(session=sess)
                                tr_hidden_state = tf.squeeze(_conv1d)
                                # fully-connected dense multiplication to transform to [8x8]
                                tr_hidden_state = tf.matmul(tf.transpose(tr_hidden_state), tr_hidden_state)
                                tr_hidden_state = tf.abs(tr_hidden_state).eval()
                            
                                start = time.time()
                                # nri_obj.adj_mat_vec = nri_obj.h_to_a(num_nodes_sc=num_nodes,
                                #                                      h_comp=tr_hidden_state, w=t_attn_W.eval())
    
                                logger.warning('time taken to generate proposals:{}: '.format(time.time() - start))
                                
                                nri_obj.adj_mat_vec = tf.random_normal(shape=(nri_obj.n_proposals, num_nodes,num_nodes))
                                # nri_obj.adj_mat_vec = tf.convert_to_tensor(nri_obj.adj_mat_vec, dtype=tf.float32)
                                
                                start_b = time.time()
                                for k in range(10):
                                    # hidden_state_tmp = tf.matmul(nri_obj.adj_mat_vec[k], hidden_state).eval()
                                    pred_path_np, prob_mat = \
                                        sess.run([krnl_mdl.pred_path_band, krnl_mdl.cost],
                                             feed_dict={
                                                 krnl_mdl.outputs: st_embeddings,
                                                 krnl_mdl.ngh: _2dconv_in.eval(),
                                                 # krnl_mdl.hidden_states: hidden_state_tmp,
                                                 krnl_mdl.out_size: pred_path[k].shape[2],
                                                 krnl_mdl.pred_path_band: pred_path[k]
                                             })

                                    pred_path[k] = sess.run(krnl_mdl.pred_path_band)
                                rcmndr_start = time.time()
                                
                                euc_min, fde_min = nri_obj.assess_rcmndr(sess, out_graph,
                                                                num_nodes, batch_v.shape[0] , euc_loss,\
                                                                fde, pred_path,\
                                                                nri_obj.target_traj0_ten,
                                                                nri_obj.adj_mat_vec)

                                rcmndr_end = time.time()
                                logger.warning('\n time taken to recommend best adjacency proposal: {}\n'.format(rcmndr_end-rcmndr_start))

                                euc_min = euc_min.eval()
                                fde_min = np.min(fde_min.eval())
                                min_idx = np.argmin(euc_min)

                                euc_min = np.min(euc_min)
                                # fde_min = np.min(fde_min)
                                bst_adj_prop = nri_obj.adj_mat_vec[min_idx]
                                # fde_min = fde_min.eval()
                                # bst_adj_prop = bst_adj_prop.eval()[0]
                                
                                print('L2-Loss = ', euc_min)
                                print('FDE = ', fde_min)
                                
                                nri_obj.l2norm_vec = tf.Variable(tf.convert_to_tensor(euc_min))
    
                                # TODO: think about optimizing at each prediction or optimizing after batch
                                start = time.time()
                                
                                nri_obj.loss_optzr.run(session=sess)
                                logger.warning('BackPropagation with SGD took:{}'.format(time.time()-start))
                                
                                # adj_mat = nri_obj.select_best_rlns(n_adj=nri_obj.adj_mat_vec, g=graph_t)
                                # weigh hidden states then send them back to glstm on next step
                                # weightage of final hidden states resulting from chaining of hidden states through social (dynamic) neighborhood then weighted static neighborhood
                                # then Softmaxing hidden states
                                # bst_adj_prop = nri_obj.adj_mat_vec[bst_adj_prop]
                                # tf.pad(bst_adj_prop, [[1, 1], [1, 1]])
                                krnl_mdl.hidden_states = tf.matmul(bst_adj_prop, hidden_state)
                                krnl_mdl.hidden_states = tf.nn.softmax(krnl_mdl.hidden_states)
                                # make it stateful when batch_size is small enough to find trajectories
                                # related to each other between 2 consecutive batches .
                                hidden_state = krnl_mdl.hidden_states.eval()
                                
                                # krnl_mdl.attn = tf.Variable(krnl_mdl.attn)
                                
                                # pred_path = np.transpose(pred_path.eval(), (2, 1, 0))
                                num_targets += num_nodes
                                # sess.run(loss, feed_dict={
                                #     l2norm_vec: tf.cumsum(euc_loss)/(num_nodes*12)})
    
                                print('\nTime taken evaluate 10 predictions per each pedestrian + Social recommendation = ',
                                      time.time() - start_b)
                                time_log.write('{0},{1},{2},{3}\n'.format(e, b, time.time() - start_b,
                                               (pid.memory_info().rss / 1024 / 1024 / 1024)))
                                
                                log_dir.write('{0},{1},{2}\n'.format(e, b, euc_min))
                                log_dir_fde.write('{0},{1},{2}\n'.format(e, b, fde_min))
                                # if b % 10 == 0:
                                #     # Try here
                                #     tf.reset_default_graph()
                                #     del out_graph
                                #     out_graph = tf.Graph()

                                if b == 32:
                                    log_count_f.close()
                                    log_dir.close()
                                    log_dir_fde.close()
                                    sess.close()
                                    out_sess.close()
                                    return
                                # End of batch
                        
                            print('============================')
                            print("Memory used: {:.2f} GB".format(pid.memory_info().rss / 1024 / 1024 / 1024))
                            print('============================')
                        
                            
                            # tf.reset_default_graph()
                            batch, target_traj, _ = dataloader.next_step()
    
                            graph_t = graph.ConstructGraph(current_batch=batch, framenum=frame,
                                                           future_traj=target_traj)
                            batch_v = list(graph_t.get_node_attr(param='node_pos_list').values())
    
                            if len(batch_v) == 0:
                                break
    
                            if len(np.array(batch_v).shape) > 1:
                                batch_v = np.array(batch_v)[frame:frame + args.obs_len]
                                batch_v = np.linalg.norm(batch_v, axis=2).squeeze()
                            else:
                                # batch_v = np.array(batch_v)[frame:frame + args.obs_len]
                                dataloader.reset_data_pointer()
                                break
    
                            batch_v = np.transpose(batch_v)
                            try:
                                num_nodes = args.maxNumPeds # batch_v.shape[1]
                            except IndexError:
                                dataloader.reset_data_pointer()
                                break

                            with tf.variable_scope('weight_input', reuse=True):
                                # init_w = tf.initializers.random_normal(mean=0, stddev=1, seed=0, dtype=tf.float32)
                                
                                weight_i = tf.Variable(name='weight_i',
                                                       initial_value=init_w(
                                                           shape=(num_nodes, args.num_freq_blocks)),
                                                       trainable=True, dtype=tf.float32)
                                weight_ii = tf.Variable(name='weight_ii',
                                                        initial_value=init_w(
                                                            shape=(dim, args.obs_len)),
                                                        trainable=True, dtype=tf.float32)
                                
                            tf.initialize_variables(var_list=[weight_i, weight_ii]).run()
                            
                            tf.initialize_variables(
                                var_list=[krnl_mdl.weight_v, krnl_mdl.bias_v]).run()
    
                            tf.initialize_variables(
                                var_list=[krnl_mdl.cost, krnl_mdl.weight_c,
                                          krnl_mdl.weight_o]).run()
    
                            end_t = time.time()
                            logger.warning('{0} seconds to complete'.format(end_t - start_t))
                            logger.warning('Frame {3} Batch {0} of {1}, Loss = {2}, num_ped={4}'
                                           .format(b, dataloader.num_batches, krnl_mdl.cost, frame, len(target_traj)))
    
                            if (e * dataloader.num_batches + b) % args.save_every == 0: #e % args.save_every == 0 and
                                logger.warning('Saving model at batch {0}, epoch {1}'.format(b, e))
                                checkpoint_path = os.path.join(save_dir,
                                                               'aol_osc_train_{0}_{1}_{2}_{3}.ckpt'
                                                               .format(d, e, b, dataloader.sel))

                                try:
                                    sess.run(fetches=tf.initialize_all_variables()) #critical when it initializes all variables stored in default graph each time
                                except tf.errors.FailedPreconditionError:
                                    sess.run(fetches=tf.initialize_all_variables())
                                saver = tf.train.Saver(tf.all_variables())
                                saver.save(sess, checkpoint_path,
                                           global_step=e * dataloader.num_batches + b)
    
                                logger.warning("model saved to {}".format(checkpoint_path))
                        
                        e += 1
                        e_end = time.time()
        
                        logger.warning('Epoch time taken: {}'.format(e_end - e_start))
                        log_count_f.write(
                            'Dataset {0}= ADE steps {1}\nFDE steps = {2}'.format(d, num_targets, num_end_targets))
            # log_f.close()
            sess.close()
            time_log.close()
            log_count_f.close()
            log_dir.close()
            log_dir_fde.close()
        out_sess.close()
        del(graph)
        
if __name__ == '__main__':
    main()
