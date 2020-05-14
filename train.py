import seaborn as sb
import argParser as parser
import time
import sys
import psutil
import logging
from models import g2k_lstm_mcr as mcr
# from models import gsk_lstm_cell as cell
# from models import g2k_lstm_mc as MC
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

            # dataloader = load.DataLoader(args=args, datasets=[0, 1, 2, 3, 4, 5], start=2, sel=0)

            datasets = {5}
            # datasets.remove(l)  # leave dataset

            #'/home//Documents/PycharmProjects/AOL/'
            for d in datasets:  # range(2,5):
                # /home/siri0005/Documents/multimodaltraj_2/log/

                dataloader = load.DataLoader(args=args, datasets=[0, 1, 2, 3, 4, 5, 6], start=d, sel=0)
                parent_dir = dataloader.parent_dir
                time_log = open(os.path.join(parent_dir, dataloader.used_data_dirs[d], 'training_Tlog.txt'), 'w')
                log_count = parent_dir+'/log/aol_ovsc_counts_{0}.txt'.format(
                    d)
                log_count_f = open(log_count, 'w')
                log_dir = open(parent_dir+'/log/aol_ovsc_ade_log_kfold_{0}.csv'.format(
                    d), 'w')
                log_dir_fde = open(parent_dir+'/log/aol_ovsc_fde_log_kfold_{0}.csv'.format(
                    d),'w')

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
    
                            # tf.keras.layers.Conv1D(filters=1, kernel_size=73, input_shape=((16, 80, 1)),
                            #                              kernel_initializer='he_normal',
                            #                              use_bias=False,
                            #                              strides=1, activation='relu', padding='valid')
    
                            ctxt_img_path = glob.glob(dataloader.current_dir + 'ctxt.png')
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

                            # tf.initialize_all_variables()
    
                            # stat_ngh = helper.neighborhood_stat_enc(
                            #     hidden_size=args.rnn_size,
                            #     num_layers=args.num_layers,
                            #     grid_size=args.grid_size,
                            #     dim=dim,
                            #     ctxt_path=ctxt_img_path)
                            # with tf.variable_scope('ngh_stat'):
                            #     static_mask = tf.placeholder(name='static_mask',  # shape=(dim, static_frame_w),
                            #                                  dtype=tf.float32)
                            #     social_frame = tf.placeholder(name='social_frame',  # shape=(static_frame_w,dim),
                            #                                   dtype=tf.float32)
                            #     state_f00_b00_c = tf.placeholder(name='state_f00_b00_c',  # shape=(dim,hidden_size),
                            #                                      dtype=tf.float32)
                            #     c_hidden_states = tf.placeholder(name='c_hidden_states',
                            #                                      # shape=(dim, (grid_size * (grid_size/2))),
                            #                                      dtype=tf.float32)
                            #     output = tf.placeholder(dtype=tf.float32,
                            #                             # shape=[num_nodes, (grid_size * (grid_size / 2))],
                            #                             name="output")
    
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
                            
                            # nri_obj.target_traj0_ten = nri_obj.target_traj0_ten[0]
    
                            # if int(nri_obj.target_traj0_ten.shape[0]) < num_nodes:
                            #     pred_path = sess.run([pred_path], {pred_path:tf.random_normal(shape=(10, 2, 12, \
                            #                                         int(nri_obj.target_traj0_ten.shape[0])),
                            #                                  dtype=tf.float32).eval()})[0]
                            # else:
                            pred_path = sess.run([pred_path], {pred_path: tf.random_normal(shape=(10, 2, 12, num_nodes),
                                                               dtype=tf.float32).eval()})[0]
                            
                            krnl_mdl = mcr.g2k_lstm_mcr(in_features=nghood_enc.input,  # MC.g2k_lstm_mc
                                                        num_nodes=pred_path.shape[3],
                                                        obs_len=args.obs_len,
                                                        hidden_size=args.rnn_size,
                                                        lambda_reg=args.lambda_param,
                                                        sess_g=out_graph)
    
                            vislet = np.array(dataloader.vislet[:, frame:frame + num_nodes], dtype=np.float32)
    
                            with tf.variable_scope('weight_input',reuse=True):
                                
                                weight_i = tf.Variable(name='weight_i',
                                                           initial_value=init_w(
                                                               shape=(rl_num_nodes, args.num_freq_blocks)),
                                                       trainable=True, dtype=tf.float32)
                                weight_ii = tf.Variable(name='weight_ii',
                                                        initial_value=init_w(
                                                            shape=(dim, args.obs_len)),
                                                        trainable=True, dtype=tf.float32)
                                weight_vi = tf.Variable(name='weight_vi',
                                                        initial_value=init_w(
                                                            shape=(vislet.shape[1], args.num_freq_blocks)),
                                                        trainable=True, dtype=tf.float32)
    
                            tf.initialize_variables(var_list=[weight_i, weight_ii, weight_vi]).run()
    
                            inputs = tf.convert_to_tensor(batch_v, dtype=tf.float32)
                            inputs = tf.matmul(inputs, weight_i)
                            inputs = tf.matmul(weight_ii, inputs)
    
                            # Embed features set into fixed-shaped compact tensor [8x8]
                            vislet_emb = tf.matmul(vislet, weight_vi)
                            F = np.concatenate((inputs.eval(), vislet_emb.eval()))
    
                            tf.initialize_variables(
                                var_list=[krnl_mdl.weight_r, krnl_mdl.weight_v, krnl_mdl.bias_v]).run()  # , krnl_mdl.embed_vis
                            tf.initialize_variables(var_list=[krnl_mdl.cost, krnl_mdl.attn, krnl_mdl.weight_c,
                                                              krnl_mdl.weight_o, nri_obj.l2norm_vec]).run() # krnl_mdl.attn,
                            
                            # sess.run(fetches=tf.initialize_all_variables())
    
                            start_t = time.time()
                            vislet_past = vislet_emb
                            vislet_rel = vislet_past * vislet_emb
    
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
                                             feed_dict={nghood_enc.input: F,
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
    
                                krnl_attn = krnl_mdl.attn.eval()
                                t_attn_W = tf.Variable(tf.nn.softmax(tf.exp(krnl_attn) / tf.cumsum(tf.exp(krnl_attn))))
                                tf.initialize_variables([t_attn_W]).run()
                                
                                # logging.warning('NMF components W shape:{}'.format(tr_hidden_state.shape))
                                # logging.warning('NMF features H shape:{}'.format(t_attn_W.shape))
                                # logging.warning('number of nodes created:{}'.format(num_nodes))
                                
                                start = time.time()
                                nri_obj.adj_mat_vec = nri_obj.h_to_a(num_nodes_sc=num_nodes,
                                                                     h_comp=tr_hidden_state, w=t_attn_W.eval())
    
                                logger.warning('time taken to generate proposals:{}: '.format(time.time() - start))
                                
                                nri_obj.adj_mat_vec = tf.convert_to_tensor(nri_obj.adj_mat_vec, dtype=tf.float32)
                                
                                start_b = time.time()
                                for k in range(10):
                                    # tf.pad(nri_obj.adj_mat_vec[k], paddings=[[1,1], [1,1]])
                                    hidden_state_tmp = tf.matmul(nri_obj.adj_mat_vec[k], hidden_state).eval()
                                    pred_path_np, hidden_state, prob_mat = \
                                        sess.run([krnl_mdl.pred_path_band, krnl_mdl.hidden_states, krnl_mdl.cost],
                                             # krnl_mdl.hidden_states,
                                             feed_dict={
                                                 krnl_mdl.outputs: #st_embeddings,
                                                 np.concatenate((st_embeddings, vislet_emb.eval()), axis=0),
                                                 krnl_mdl.ngh: _2dconv_in.eval(),
                                                 krnl_mdl.rel_features: vislet_rel.eval(),
                                                 krnl_mdl.hidden_states: hidden_state_tmp,
                                                 krnl_mdl.out_size: pred_path[k].shape[2],
                                                 # krnl_mdl.lambda_reg: args.lambda_reg,
                                                 krnl_mdl.pred_path_band: pred_path[k]
                                             })
    
                                    # tf.assign(ref=pred_path[k], value=tf.convert_to_tensor(pred_path_np), use_locking=False)
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
    
                            # vislet = dataloader.vislet[:, frame:frame + num_nodes]
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
                                weight_vi = tf.Variable(name='weight_vi',
                                                        initial_value=init_w(
                                                            shape=(vislet.shape[1], args.num_freq_blocks)),
                                                        trainable=True, dtype=tf.float32)
    
                            tf.initialize_variables(var_list=[weight_i, weight_ii, weight_vi]).run()
                            
                            tf.initialize_variables(
                                var_list=[krnl_mdl.weight_v,krnl_mdl.weight_r, krnl_mdl.bias_v]).run()
    
                            tf.initialize_variables(
                                var_list=[krnl_mdl.cost, krnl_mdl.attn, krnl_mdl.weight_c,
                                          krnl_mdl.weight_o]).run()
    
                            end_t = time.time()
                            logger.warning('{0} seconds to complete'.format(end_t - start_t))
                            logger.warning('Frame {3} Batch {0} of {1}, Loss = {2}, num_ped={4}'
                                  .format(b, dataloader.num_batches, krnl_mdl.cost, frame, len(target_traj)))
    
                            # sess.run(train)
    
                            if (e * dataloader.num_batches + b) % args.save_every == 0: #e % args.save_every == 0 and
                                logger.warning('Saving model at batch {0}, epoch {1}'.format(b, e))
                                checkpoint_path = os.path.join(parent_dir+'/save/ovsc', \
                                                               'aol_ovsc_train_{1}_{0}_{2}.ckpt'.format(e, d, b))
    
                                try:
                                    sess.run(fetches=tf.initialize_all_variables()) #critical when it initializes all variables stored in default graph each time
                                except tf.errors.FailedPreconditionError:
                                    sess.run(fetches=tf.initialize_all_variables())
                                saver = tf.train.Saver(tf.all_variables())
                                saver.save(sess, checkpoint_path,
                                           global_step=e * dataloader.num_batches + b)
    
                                logger.warning("model saved to {}".format(checkpoint_path))
                        # make another model file with attn
                        # fde_np = np.stack(fde)
                        # euc_loss_np = np.ravel(euc_loss)
                        # np.savetxt(fname=log_dir_fde, X=fde_np, delimiter=",")
                        # np.savetxt(fname=log_dir, X=euc_loss_np, delimiter=",")
                        # /home/siri0005/Documents/multimodaltraj_2/save
                        # log_f.write('{0}'.format(euc_loss))
                        e += 1
                        # out_sess.graph = None
                        # out_sess.graph = out_graph
                        # del(graph)
                        # graph = nx_g.online_graph(args)
    
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
    # else:
    # *************************************************************** VALIDATION *************************************
    # Validate

    # out_graph = tf.Graph()
    # with out_graph.as_default():
    #     with tf.Session(graph=out_graph).as_default() as val_sess:
    #         graph = nx_g.online_graph(args)
    #         l = args.leaveDataset
    #         dataloader = load.DataLoader(args=args, datasets=[0, 1, 2, 3, 4, 5], start=l, sel=0)
    #         dataloader.reset_data_pointer(valid=True)
    #         parent_dir = '/home/phd-2020/Documents/PycharmProjects/multimodaltraj_2/'
    #         # out_sess.close()
    #         # with tf.Session(graph=out_graph).as_default() as sess:
    #         #int(input("Select which dataset to validate: 2.Zara1, 3.Zara2, 4.UCY \n"))
    #         checkpoint_path = parent_dir+'save'
    #         ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_path, latest_filename='checkpoint')
    #
    #         # process epoch number *****************************************
    #         import re
    #         e = re.findall(r'_[0-9]+', ckpt.model_checkpoint_path)[1]
    #         e = int(e.replace('_', ''))
    #         # **************************************************************
    #
    #         true_path = []
    #         frame = 1
    #
    #         print('Importing meta data of stored model in' + ckpt.model_checkpoint_path + '.meta')
    #         saver = tf.train.import_meta_graph(
    #             os.path.join(checkpoint_path, ckpt.model_checkpoint_path + '.meta'))
    #         # print ('loading model: ', ckpt.model_checkpoint_path)
    #
    #         print('loading model: ', ckpt.all_model_checkpoint_paths)
    #
    #         # Restore the model at the checkpoint
    #         saver.restore(val_sess, ckpt.model_checkpoint_path)
    #         # def_graph = tf.get_default_graph()
    #         dim = int(args.neighborhood_size / args.grid_size)
    #         dataloader = load.DataLoader(args=args, datasets=[0, 1, 2, 3, 4, 5], start=l, sel=0, infer=True)
    #         traj = dataloader.load_trajectories(data_file=dataloader.sel_file)
    #
    #         dataloader.valid_frame_pointer = int((dataloader.len - int(
    #             dataloader.max * .7)) / dataloader.val_max)  # + dataloader.seq_length #+ (frame * e)
    #
    #         dataloader.valid_num_batches = int(dataloader.val_max / (dataloader.batch_size* 20)) #
    #
    #         with tf.variable_scope('nghood_init', reuse=True):
    #             out_init = tf.zeros(dtype=tf.float32, shape=(
    #                 dim, dim))  # (args.grid_size * (args.grid_size / 2))))
    #             c_hidden_init = tf.zeros(dtype=tf.float32, shape=(
    #                 dim, dim))  # (args.grid_size * (args.grid_size / 2))))
    #
    #         # variables = def_graph.get_collection('variables')
    #         # val_traj = [{idx: traj[idx]} for idx in dataloader.trajectories]
    #         # frame_iter = iter(val_traj)
    #         # dataloader.frame_pointer = dataloader.valid_frame_pointer
    #         cv_ade_err = []
    #         cv_fde_err = []
    #
    #         dim = int(args.neighborhood_size / args.grid_size)
    #
    #         for vb in range(dataloader.valid_num_batches):
    #             # for vb in range(5):
    #             # Get batch data
    #             # x, _, d = dataloader.load_trajectories(data_file='')  ## stateless lstm without shuffling
    #             # rang = range(dataloader.valid_frame_pointer, dataloader.valid_frame_pointer + int(dataloader.raw_data.shape[1]*0.3), args.seq_length)
    #             # for idx in rang:
    #             #     try:
    #             #         val_traj[idx] = dataloader.raw_data[0:2,idx]
    #             #     except KeyError:
    #             #         continue
    #             num_targets = 0
    #             start_t = time.time()
    #             batch, target_traj, fp = dataloader.next_step()
    #
    #             if len(batch) == 0:
    #                 break
    #
    #             graph_t = graph.ConstructGraph(current_batch=batch, framenum=fp, future_traj=target_traj)
    #             batch_v = list(graph_t.get_node_attr(param='node_pos_list').values())
    #
    #             if len(batch_v) == 0:
    #                 break
    #
    #             if len(np.array(batch_v).shape) > 1:
    #                 batch_v = np.array(batch_v)[frame:frame + args.obs_len]
    #                 batch_v = np.linalg.norm(batch_v, axis=2).squeeze()
    #             else:
    #                 # batch_v = np.array(batch_v)[frame:frame + args.obs_len]
    #                  break
    #
    #             batch_v = np.transpose(batch_v)
    #             num_nodes = batch_v.shape[1]
    #
    #             if vb == 0:
    #                 init_w = tf.initializers.random_normal(mean=0, stddev=1, seed=0, dtype=tf.float32)
    #                 hidden_state = np.zeros(shape=(dim, args.rnn_size))
    #                 nghood_enc = helper.neighborhood_vis_loc_encoder(
    #                     hidden_size=args.rnn_size,
    #                     hidden_len=dim,
    #                     num_layers=args.num_layers,
    #                     grid_size=args.grid_size,
    #                     embedding_size=args.embedding_size,
    #                     dropout=args.dropout)
    #
    #                 ctxt_img_path = glob.glob(dataloader.current_dir + 'ctxt.png')
    #                 ctxt_img = tf.convert_to_tensor(imread(ctxt_img_path[0]), dtype=tf.float32)
    #
    #                 ctxt_img_pd = tf.convert_to_tensor(
    #                     tf.pad(ctxt_img, paddings=tf.constant([[1, 1, ], [0, 1], [0, 0]])),
    #                     dtype=tf.float32)
    #
    #                 ctxt_img_pd = tf.expand_dims(ctxt_img_pd, axis=0)
    #                 _2dconv = tf.nn.conv2d(input=ctxt_img_pd,
    #                                        filter=
    #                                        tf.random_normal(shape=[1, 1, 3, 1], dtype=tf.float32),
    #                                        padding='VALID', strides=[1, 1, 1, 1])
    #
    #                 _2dconv = tf.transpose(tf.squeeze(_2dconv))#.eval(session=val_sess)
    #                 width = int(_2dconv.shape.dims[0])
    #                 height = int(_2dconv.shape.dims[1])
    #
    #                 # _2dconv = args.lambda_param * _2dconv
    #                 # _2dconv = tf.nn.conv2d(input=ctxt_img_pd,
    #                 #                        filter=tf.random_normal(shape=[1, 1, 3, 1], dtype=tf.float32),
    #                 #                        padding='VALID', strides=[1, 1, 1, 1])
    #                 #
    #                 # _2dconv = tf.squeeze(_2dconv)
    #                 # stat_mask = tf.random_normal(shape=(height, args.rnn_size), dtype=tf.float32)
    #                 # _2dconv_in = tf.matmul(_2dconv, stat_mask)
    #                 # adj_mat = sk.block_reduce(_2dconv_in.eval(), (int(height / args.obs_len), args.rnn_size),
    #                 #                           np.average)
    #                 # sb.heatmap(adj_mat)
    #                 # py.show()
    #
    #                 stat_ngh = helper.neighborhood_stat_enc(
    #                     hidden_size=args.rnn_size,
    #                     num_layers=args.num_layers,
    #                     grid_size=args.grid_size,
    #                     dim=dim)
    #
    #                 with tf.variable_scope('ngh_stat'):
    #                     static_mask = tf.placeholder(name='static_mask',  # shape=(dim, static_frame_w),
    #                                                  dtype=tf.float32)
    #                     social_frame = tf.placeholder(name='social_frame',  # shape=(static_frame_w,dim),
    #                                                   dtype=tf.float32)
    #                     state_f00_b00_c = tf.placeholder(name='state_f00_b00_c',  # shape=(dim,hidden_size),
    #                                                      dtype=tf.float32)
    #                     c_hidden_states = tf.placeholder(name='c_hidden_states',
    #                                                      # shape=(dim, (grid_size * (grid_size/2))),
    #                                                      dtype=tf.float32)
    #                     output = tf.placeholder(dtype=tf.float32,
    #                                             # shape=[num_nodes, (grid_size * (grid_size / 2))],
    #                                             name="output")
    #             # krnl_mdl = MC.g2k_lstm_mc(in_features=nghood_enc.input,
    #             #                             num_nodes=num_nodes, obs_len=args.obs_len,
    #             #                             lambda_reg=args.lambda_param,
    #             #                             sess_g=def_graph)
    #
    #             krnl_mdl = mcr.g2k_lstm_mcr(in_features=nghood_enc.input,
    #                                         num_nodes=num_nodes, obs_len=args.obs_len,
    #                                         hidden_size=args.rnn_size,
    #                                         lambda_reg=args.lambda_param,
    #                                         sess_g=out_graph)
    #
    #             # num_targets += num_nodes
    #             # krnl_mdl = MC.g2k_lstm_mc(in_features=nghood_enc.input,
    #             #                             num_nodes=num_nodes, obs_len=args.obs_len,
    #             #                             lambda_reg=args.lambda_param,
    #             #                             # hidden_size=args.rnn_size,
    #             #                             sess_g=out_graph)
    #
    #             with tf.variable_scope('weight_input', reuse=True):
    #                 weight_i = tf.Variable(name='weight_i',
    #                                        initial_value=init_w(shape=(num_nodes, dim)),
    #                                        trainable=True, dtype=tf.float32)
    #                 weight_ii = tf.Variable(name='weight_ii',
    #                                         initial_value=init_w(shape=(dim, args.obs_len)),
    #                                         trainable=True, dtype=tf.float32)
    #
    #             tf.initialize_variables(var_list=[weight_i, weight_ii]).run()
    #
    #             inputs = tf.convert_to_tensor(batch_v, dtype=tf.float32)
    #             inputs = tf.matmul(inputs, weight_i)
    #             inputs = tf.matmul(weight_ii, inputs)
    #
    #             vislet = dataloader.vislet[:,
    #                      dataloader.valid_frame_pointer:dataloader.valid_frame_pointer + num_nodes]  # tf.expand_dims(batch_v[0], axis=0)
    #             vislet_emb = tf.matmul(vislet, weight_i)
    #
    #             if vb == 0:
    #                 vislet_past = vislet_emb
    #
    #             vislet_rel = vislet_past * vislet_emb
    #
    #             stat_mask = tf.random_normal(shape=(args.obs_len, args.obs_len), dtype=tf.float32) #shape=(height, args.rnn_size)
    #             # stat_mask += tf.expand_dims(tf.range(start=0, limit=1, delta=(1 / args.obs_len), dtype=tf.float32), axis=0)
    #             # _2dconv_in = tf.matmul(_2dconv, stat_mask)
    #             # _2dconv_in = sk.block_reduce(_2dconv_in.eval(session=val_sess), (int(width/args.obs_len), int(args.rnn_size / args.obs_len)), np.average)
    #             static_mask_nd = stat_mask.eval() #_2dconv_in
    #
    #             tf.initialize_variables(
    #                 var_list=[krnl_mdl.weight_r,krnl_mdl.weight_v, krnl_mdl.bias_v]).run()  # ,krnl_mdl.embed_vis
    #             tf.initialize_variables(
    #                 var_list=[krnl_mdl.attn,krnl_mdl.cost, krnl_mdl.weight_c,
    #                           krnl_mdl.weight_o]).run()
    #
    #             # for frame_iter in iter(val_traj):
    #             # check if get_node_attr gets complete sequence for all nodes
    #             # num_nodes x obs_length
    #             # if num_nodes > 0 :
    #             cv_err = []
    #             fde = []
    #
    #             # (frame, _), = frame_iter.items()
    #             for frame in batch:
    #                 print('Frame {0}'.format(frame))
    #                 true_path.append(batch[frame])
    #                 # except KeyError:
    #                 #     if frame == max(batch.keys()):
    #                 #         break
    #                 #     elif frame + args.seq_length + 1 > max(batch.keys()):
    #                 #         frame = list(batch.keys())[0]
    #                 #     else:
    #                 #         frame += args.seq_length + 1
    #                 #     continue
    #                 # with tf.variable_scope('ngh_stat'):
    #                 #     static_mask = tf.placeholder(name='static_mask',  # shape=(dim, static_frame_w),
    #                 #                                  dtype=tf.float32)
    #                 #
    #                 #     social_frame = tf.placeholder(name='social_frame',  # shape=(static_frame_w,dim),
    #                 #                                   dtype=tf.float32)
    #                 #     state_f00_b00_c = tf.placeholder(name='state_f00_b00_c',  # shape=(dim,hidden_size),
    #                 #                                      dtype=tf.float32)
    #                 #     c_hidden_states = tf.placeholder(name='c_hidden_states',
    #                 #                                      # shape=(dim, (grid_size * (grid_size/2))),
    #                 #                                      dtype=tf.float32)
    #                 #
    #                 #     output = tf.placeholder(dtype=tf.float32,
    #                 #                             # shape=[num_nodes, (grid_size * (grid_size / 2))],
    #                 #                             name="output")
    #                 start = time.time()
    #                 st_embeddings, hidden_state, ng_output, c_hidden_state = \
    #                     val_sess.run([nghood_enc.input, nghood_enc.state_f00_b00_c,
    #                               nghood_enc.output, nghood_enc.c_hidden_state],
    #                              feed_dict={nghood_enc.input: inputs.eval(session=val_sess),
    #                                         nghood_enc.state_f00_b00_c: hidden_state,
    #                                         nghood_enc.output: out_init.eval(session=val_sess),
    #                                         nghood_enc.c_hidden_state: c_hidden_init.eval(session=val_sess)})
    #
    #                 static_mask_np, social_frame_np =\
    #                     val_sess.run([static_mask, output],
    #                              feed_dict={static_mask: static_mask_nd,
    #                                         social_frame: ng_output,
    #                                         state_f00_b00_c: hidden_state,
    #                                         output: out_init.eval(),
    #                                         c_hidden_states: c_hidden_init.eval()
    #                                         })
    #
    #                 input = tf.matmul(b=static_mask_np,
    #                                   a=social_frame_np).eval()  # Soft-attention mechanism equipped with static grid
    #                 combined_ngh, hidden_state = val_sess.run([stat_ngh.input, stat_ngh.hidden_state],
    #                                                       feed_dict={stat_ngh.input: input, #_2dconv_in, #
    #                                                                  stat_ngh.hidden_state: hidden_state})
    #
    #                 # reg_ng = np.transpose(args.lambda_param * np.transpose(ng_output))
    #                 # hidden_state,
    #
    #                 pred_path = tf.random_normal(shape=(2, 12, num_nodes))
    #                 pred_path, prob_mat = \
    #                     val_sess.run([krnl_mdl.pred_path_band, krnl_mdl.cost],
    #                              # krnl_mdl.hidden_states,
    #                              feed_dict={
    #                                  krnl_mdl.outputs:  # st_embeddings,
    #                                     np.concatenate((st_embeddings, vislet_emb.eval(session=val_sess)), axis=0),
    #                                  krnl_mdl.ngh: ng_output, #_2dconv_in
    #                                  krnl_mdl.rel_features: vislet_rel.eval(session=val_sess),
    #                                  krnl_mdl.hidden_states: hidden_state,
    #                                  krnl_mdl.out_size: num_nodes,
    #                                  # krnl_mdl.lambda_reg: args.lambda_reg,
    #                                  krnl_mdl.pred_path_band: pred_path.eval(session=val_sess)
    #                              })
    #
    #                 end_t = time.time()
    #                 print('Prediction took {0} seconds to complete\n'.format(end_t - start))
    #                 # krnl_mdl.ngh = reg_ng
    #                 # krnl_mdl.out_size = tf.Variable(num_nodes)
    #                 # krnl_mdl.hidden_states = hidden_state
    #                 # krnl_mdl.outputs = np.concatenate((st_embeddings, vislet_emb.eval()), axis=0)
    #                 # krnl_mdl.pred_path_band = tf.random_normal(shape=(2, 12, num_nodes))
    #                 # krnl_mdl.rel_features = vislet_rel.eval(session=val_sess)
    #                 # krnl_mdl.forward()
    #                 # pred_path = krnl_mdl.pred_path_band
    #                 # attn = tf.nn.softmax(tf.exp(krnl_mdl.attn) / tf.cumsum(tf.exp(krnl_mdl.attn)))
    #                 # weigh hidden states then send them back to glstm on next step
    #                 # krnl_mdl.hidden_states = tf.nn.softmax(krnl_mdl.hidden_states)
    #                 # hidden_state = krnl_mdl.hidden_states.eval()
    #                 # weightage of final hidden states resulting from chaining of hidden states through social (dynamic) neighborhood then weighted static neighborhood
    #                 # then softmaxing hidden states
    #                 # hidden_state = np.matmul(attn.eval(), hidden_state)
    #                 # adj_mat = tf.matmul(tf.nn.softmax(hidden_state),
    #                 #                     tf.ones(shape=(hidden_state.shape[1], 1), dtype=tf.float32))
    #                 # GG-NN 2016 A_nx2n , we use A_nxn; n is |G_v| cardinality of node v in Graph G.
    #                 # hidden_state = adj_mat.eval() * hidden_state
    #                 attn = tf.nn.softmax(tf.exp(krnl_mdl.attn) / tf.cumsum(tf.exp(krnl_mdl.attn)))
    #
    #                 # weigh hidden states then send them back to glstm on next step
    #                 krnl_mdl.hidden_states = tf.nn.softmax(krnl_mdl.hidden_states)
    #                 hidden_state = krnl_mdl.hidden_states.eval(session=val_sess)
    #                 # # weightage of final hidden states resulting from chaining of hidden states through social (dynamic) neighborhood then weighted static neighborhood
    #                 # # then softmaxing hidden states
    #                 hidden_state = tf.matmul(attn, hidden_state).eval()
    #                 adj_mat = sk.block_reduce(hidden_state, (1, int(args.rnn_size/args.obs_len)), np.max)
    #                 # adj_mat = tf.nn.max_pool(hidden_state, ksize=(8,16), strides=(1,1), data_format="NHWC", padding='VALID')
    #                 # tf.matmul(tf.nn.softmax(hidden_state),
    #                 # tf.random_normal(shape=(hidden_state.shape[1], hidden_state.shape[0]), dtype=tf.float32))
    #                 # # GG-NN 2016 A_nx2n , we use A_nxn; n is |G_v| cardinality of node v in Graph G.
    #                 hidden_state = tf.matmul(adj_mat, hidden_state).eval(session=val_sess)
    #                 pred_path = np.transpose(pred_path, (2, 1, 0))
    #
    #                 if frame == 8015:
    #                     xy = adj_mat*10 #.eval()
    #                     # xy = xy >= (np.max(xy)*.5)
    #                     # # x, y = xy[:, 0], xy[0, :]
    #                     # py.show()
    #                     # fig = py.figure()
    #                     # plt = fig.add_subplot(111)
    #                     # py.savefig(
    #                     #     fname='/home/phd-2020/Documents/PycharmProjects/multimodaltraj_2/plots/zara1_6017_htmap.png')
    #                     #
    #                     # heatmap,xedges,yedges = sb.heatmap(xy[0,:]) np.histogram2d(x, y, bins=(8, 1))
    #                     # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    #                     # py.imshow(heatmap, extent=extent, shape=(8, 8), alpha=0.7)
    #                     # py.xticks([])
    #                     # py.yticks([])
    #
    #                     # py.plot(pred_path[i][:, 0], pred_path[i][:, 1], 'r*-')
    #                     # py.plot()
    #                     heatmap = sb.heatmap(xy)
    #                     py.show()
    #
    #                 # ped_ids = list(graph_t.getNodes().keys())
    #                 num_targets += num_nodes
    #                 if num_nodes > 0:
    #                     for i, itr in zip(range(num_nodes), iter(target_traj)):
    #                         try:
    #                             if len(target_traj[itr]) < args.pred_len:
    #                                 euc_loss = np.linalg.norm(
    #                                     (pred_path[i][0:len(target_traj[itr])] - target_traj[itr]), ord=2) / len(
    #                                     target_traj) #/ 12
    #                                 err = (pred_path[i][len(target_traj[itr]) - 1] - target_traj[itr][len(target_traj[itr]) - 1])
    #                                     # np.linalg.norm(
    #                                     # (pred_path[i][len(target_traj[itr]) - 1] - target_traj[itr][len(target_traj[itr]) - 1]),
    #                                     # ord=2)
    #                                 # fde.append(np.linalg.norm((pred_path[i][0:len(target_traj[itr])] - target_traj[itr]), ord=2))
    #                             else:
    #                                 euc_loss = np.linalg.norm(
    #                                     (pred_path[i][0:args.pred_len] - target_traj[itr][0:args.pred_len]),
    #                                     ord=2) / 12  # / len(target_traj)
    #                                 err = (pred_path[i][args.pred_len - 1] - target_traj[itr][args.pred_len - 1])
    #                                     # np.linalg.norm(
    #                                     # (pred_path[i][args.pred_len - 1] - target_traj[itr][args.pred_len - 1]),
    #                                     # ord=2)
    #
    #                             fde.append(err)
    #                             cv_err.append(euc_loss)
    #
    #                             print('GT: \n', target_traj[itr][0:args.pred_len])
    #                             print('\n Pred Traj: \n' , pred_path[i])
    #                             print('****************************************************')
    #                             print('Batch = ', vb, ' Frame = ', frame, '  euc_loss = ', euc_loss)
    #                             print('fde_err = ', err)
    #                             print('****************************************************')
    #
    #                         except KeyError:
    #                             i += 1
    #                             continue
    #
    #             # cv_fde_err.append(fde)
    #             # next(frame_iter)
    #             if len(cv_err) > 0:
    #                 cv_ade_err.append(np.mean(cv_err))
    #             if len(fde) > 0:
    #                 if l == 5:
    #                     cv_fde_err.append(np.linalg.norm(fde) / num_nodes)  # np.mean(fde)
    #                 else:
    #                     cv_fde_err.append(np.linalg.norm(fde) / len(batch))  # np.mean(fde)
    #
    #             print('Batch {0} of {1}, Loss = {2}, num_ped={3}'
    #                   .format(vb, dataloader.valid_num_batches, krnl_mdl.cost, len(target_traj)))
    #
    #             # dataloader.tick_frame_pointer()
    #             dataloader.frame_pointer = frame
    #
    #         # traj = dataloader.load_trajectories(data_file=dataloader.sel_file)
    #         # ped_ids = len(list(graph_t.getNodes().keys()))
    #         # cv_ade_err = np.divide(cv_err, ped_ids)
    #         # cv_fde_err = np.divide(fde, ped_ids)
    #
    #         print('Cross-Validation total mean error (ADE) for dataset {0} = '.format(l), np.mean(cv_ade_err))
    #         print('Cross-Validation total final error (FDE) for dataset {0} = '.format(l), np.mean(cv_fde_err))
    #         # /home/siri0005/Documents/multimodaltraj_2
    #         checkpoint_path = os.path.join(parent_dir+'save',
    #                                        'g2k_mcr_model_kfold_val_{0}.ckpt'.format(l))
    #
    #         saver.save(val_sess, checkpoint_path, global_step= e * dataloader.valid_num_batches + vb)
    #         print("model saved to {}".format(checkpoint_path))
    #     val_sess.close()

if __name__ == '__main__':
    main()