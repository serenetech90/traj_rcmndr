import numpy as np
from tensorflow import nn
import tensorflow as tf
import sklearn.decomposition as sk_dec
# from tensorflow.contrib import learn as skf
# import pathos.multiprocessing as mp
# from threading import Thread

@tf.contrib.eager.defun

# pool = mp.Pool(processes=mp.cpu_count())
def evaluate(sess, graph, params, i, i0, adj_mat_vec):
    # tf.enable_eager_execution()
    # with tf.Session().as_default() as eval_ss:
    ade_tmp, final_tmp, pred_tmp = [], [], []

    def condition(pred_len, num_nodes_sc, num_nodes,
                  target_traj, pred_path, euc_loss, fde, num_end_targets, euc_min, euc_idx, i, i0):
        # _, _, _, num_nodes, _, _, _, _, _, _ = _params
        return tf.less(i, num_nodes_sc)
    
    # def pred_itr(x):
    #     print('x = ', x)
    #     return x
    #
    # def true_itr(x):
    #     return x
    
    def inner_loop_fn(pred_len, num_nodes_sc, num_nodes, target_traj, pred_path,
                                      euc_loss, fde, num_end_targets, euc_min, euc_idx, i, i0):
        # sess = tf.InteractiveSession()
        # pred_len, attn, hidden_state, num_nodes_sc, num_nodes, target_traj, \
        #                     pred_path, euc_loss, fde, num_end_targets,euc_min, euc_idx = params
        # attn0, hidden_state0 = graph.get_tensor_by_name('attn').eval(sess),\
        # graph.get_tensor_by_name('hidden_state').eval(sess)
        # tf.Print(attn,hidden_state)
        # hidden_state.eval(session=eval_ss), attn.eval(session=eval_ss)
        # graph.get_operation_by_name('attn').eval(sess), \
        # graph.get_operation_by_name('hidden_state').eval(sess)
        # with tf.Session().as_default() as eval_ss:  # hidden_state, attn = sess.run(fetches=[hidden_state,attn])
        # if i < num_nodes_sc:
        try:
            # print('\nEntered inner_loop_fn()\n')
                # def h_to_a_call():
                # adj_mat = nri_mp.h_to_a(h_comp=hidden_state, w=attn,
                #                          adj=tf.ones(shape=(num_nodes, num_nodes)))
                # hidden_state = tf.matmul(attn, hidden_state)

                # adj_mat = tf.matmul(tf.nn.softmax(hidden_state),
                #                     tf.ones(shape=(hidden_state.shape[1], 1), dtype=tf.float32))
                # # GG-NN 2016 A_nx2n , we use A_nxn; n is |G_v| cardinality of node v in Graph G.
                # tf.add(num_end_targets, 1)
                # print('nri_learned.n_proposals', nri_learned.n_proposals)
                # print('i0 = ', int(i0))
                # k = 0
                # def ten_itr_fn(euc_min):
                    # print('recursive call, k = ', pred_itr(pred_path[k]))
                    # if k < nri_learned.n_proposals and pred_path[k].shape[0] > 0:
                    #     return tf.map_fn(pred_itr, pred_path[k]),\
                    #            tf.map_fn(true_itr, target_traj)
                    # else:
                    # i0, i1 = pred_itr(pred_path[k]), true_itr(target_traj)
                    # i0 = pred_itr(pred_path[k])
                    # i1 = true_itr(target_traj)
                    # print('i0 = ', i0)
                    # print('i1 = ', i1)
                    # for i0,i1 in zip(pred_path[k], target_traj):
                    # print('Prediction length: ', pred_path[k].shape,  '  i1= ', i1, '\n')
                    # if len(i1) < pred_len:
                    #     print('calculated Euclidean Error ADE+FDE\n')
                        # krnl_mdl.pred_path_band.eval()
                        # loss = tf.reduce_sum(tf.square(y0 - y_out))
                        # euc_loss[k] = (i0[0:i1[0].shape[0]] - i1) #),axis=0)
                        # fde[k] = (i0[i1.shape[0] - 1] - i1[-1]) #), axis=0)
                        # euc_loss = np.linalg.norm((pred_path[i0][0:len(target_traj[i0])] - target_traj[i0]), ord=2)/len(target_traj)
            # if int(target_traj.shape[0]) > 0:
            # print('calculated Euclidean Error ADE+FDE')
            euc_loss = pred_path - target_traj #), axis=0)
            fde = pred_path[:,-1,:] - target_traj[:,-1, :]#(i1[-1] - i1[-1]) #), axis=0)

            # if euc_loss[k] < euc_min:
            #     euc_min = euc_loss[k]
            #     euc_idx = i
            # else:
            #     # i0 = pred_itr(pred_path[k])
            #     # i1 = true_itr(target_traj)
            # return pred_itr(pred_path[k]), true_itr(target_traj)
            # ten_itr_fn(euc_min)
            
        except KeyError:
            print('Key Error inside loop')
            pass
        # print('i = ', euc_idx)
        i = tf.add(i, 1)
        num_end_targets = tf.add(num_end_targets, 1)
        
        # i0 += 1 #tf.add(i0, 1)
        # print('i0 = ', i0)
        # print ('target tensor = ', target_traj)
        # itr = tf.convert_to_tensor(target_traj.vocabulary_list[i0])
        return pred_len, num_nodes_sc, num_nodes,\
                                 target_traj, pred_path, euc_loss, fde, num_end_targets, \
                                 euc_min, euc_idx, i, i0

        # for i, itr in zip(range(1, num_nodes), iter(target_traj)):
        # cond = lambda i, num_nodes: i < num_nodes # and itr in range(len(target_traj))
        # body = tf.map_fn(fn=lambda params,i , itr: inner_loop_fn(params, i, itr),
        #                  elems=(params, i, itr), dtype=tf.float32, parallel_iterations=50)
        # *************************************************************************************** #
    
    # j = 0
    # j0 = itr[0]
    # tf.constant(value=0)
    # body = lambda pred_len, attn, hidden_state, num_nodes_sc, num_nodes,\
    #                                             target_traj, pred_path, euc_loss, fde, num_end_targets, euc_min, euc_idx, i, i0, itr: \
    #                                             inner_loop_fn(pred_len, attn, hidden_state, num_nodes_sc, num_nodes,
    #                                             target_traj, pred_path, euc_loss, fde, num_end_targets,euc_min, euc_idx, i, i0, itr)
    # print((params[0].graph.__repr__()) + '\n')
    pred_len, num_nodes_sc, num_nodes, target_traj, pred_path, \
                                    euc_loss, fde, num_end_targets, euc_min, euc_idx = params

    print('target_traj = ', target_traj)
    print('pred_path = ', pred_path)
    for k in range(nri_learned.n_proposals):
        p = pred_path[k]
        target_traj0_var = tf.convert_to_tensor(value=target_traj)#, shape=tf.TensorShape([None, 12, 2]))
        pred_path_var = tf.convert_to_tensor(value=p)#, shape=tf.TensorShape([None, 12, 2]))
        
        loop_vars = [pred_len, num_nodes_sc,
                     num_nodes, target_traj0_var, pred_path_var, euc_loss, fde,
                     num_end_targets, euc_min, euc_idx, i, i0]
        
        shapes = [pred_len.get_shape(), num_nodes_sc.get_shape(),
                  tf.TensorShape([]),
                  tf.TensorShape([None, 12, 2]), tf.TensorShape([None, 12, 2]),
                  euc_loss.get_shape(), fde.get_shape(), num_end_targets.get_shape(),
                  euc_min.get_shape(), euc_idx.get_shape(), i.get_shape(), i0.get_shape()]
        
        pred_len, num_nodes_sc, num_nodes, \
        _, _, euc_loss, fde, num_end_targets, \
        euc_min, euc_idx, i, i0 = \
        tf.while_loop(cond=condition, body=inner_loop_fn, loop_vars=loop_vars,
                      # shape_invariants=shapes,
                      parallel_iterations=nri_learned.n_proposals)
        
        ade_tmp.append(euc_loss)
        final_tmp.append(fde)
        # pred_tmp.append(pred_path)

    # euc_min, fde_min, fde_idx = tf.keras.backend.min(euc_loss, axis=0), tf.keras.backend.min(fde, axis=0), tf.argmin(fde)
    # inner_loop_fn(pred_len, attn, hidden_state, num_nodes_sc, num_nodes,
    #               target_traj, pred_path, euc_loss, fde, num_end_targets, euc_min, euc_idx, j, j0, itr)

    return tf.stack(ade_tmp), tf.stack(final_tmp)

def extract_ten_dict(dict, ten=[]):
    # rows_lens = []
    # due to limitations in converting to ragged tensor from list of variable-length tensors,
    # there is no other way in TF except fixing the size.
    # problem in casting type list to Tensor
    # max_ten = tf.zeros(shape=(len(dict), 12, 2))
    for itr in dict:
        if dict[itr].shape[0] < 12: # max_ten[itr].shape[0]:
            if itr < ten.shape[0]:
                ten[itr] = np.concatenate((dict[itr].eval(), np.zeros(shape=(abs(dict[itr].shape[0].value - 12), 2))), axis=0)
        # max_ten[itr] = tf.assign(max_ten[itr], (max_ten[itr] + dict[itr]))
        # rows_lens.append(int(dict[itr].shape[0]))

    return ten

class nri_learned():
     n_proposals = 10
     def __init__(self, args, sess):
        super(nri_learned, self).__init__()
        self.args = args
        self.sess = sess
        self.init = tf.random_normal(mean=0, stddev=1, seed=0, dtype=tf.float32, shape=())
        self.l2norm_vec = tf.Variable(initial_value=self.init, dtype=tf.float32, name='l2norm_vec')
        self.loss = tf.nn.l2_loss(t=self.l2norm_vec, name='loss')
        self.loss_optzr = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate).minimize(self.loss)
        self.target_traj0_ten = tf.placeholder(name='target_traj0_ten', dtype=tf.float32)
        # self.adj_mat_vec = tf.Variable(dtype=tf.float32, name='adj_mat_vec')
        # self.threaded_fn = Thread(target=self.h_to_a, daemon=True)
        # self.threaded_fn.start()

     def assess_rcmndr(self,sess, graph_t, num_nodes, batch_len, euc_loss, fde, pred_path,
                       target_traj, attn=None, hidden_state=None, adj_mat_vec=None):
            # target_traj0 = tf.feature_column.categorical_column_with_vocabulary_list(key='ID', vocabulary_list=target_traj)
            # Transform all variables to a hashable format (tf tensors) that can be iterable in the eager_execution mode
            # inside while_loop() call
            # itr = tf.convert_to_tensor(target_traj0.vocabulary_list)
            # target_traj0_ten = []
            
            pred_path = np.transpose(pred_path, (0, 3, 2, 1))

            i0 = tf.zeros(shape=())
            pred_len0 = tf.convert_to_tensor(value=self.args.pred_len)
            i = tf.zeros(shape=(), dtype=tf.float32)
            num_nodes0 = tf.convert_to_tensor(num_nodes, dtype=tf.float32)
            # hidden_state0 = tf.convert_to_tensor(hidden_state)
            num_end_targets = tf.zeros(shape=())
            euc_min = tf.convert_to_tensor(np.inf, dtype=tf.float32)
            euc_idx = tf.zeros(shape=(1), dtype=tf.float32)
            # attn0 = tf.convert_to_tensor(attn)

            euc_loss = np.zeros((num_nodes, self.args.pred_len, 2), dtype=np.float32)
            fde = np.zeros((num_nodes, 2), dtype=np.float32)

            euc_loss0 = tf.convert_to_tensor(euc_loss)
            fde0 = tf.convert_to_tensor(fde)

            params = [pred_len0, num_nodes0, num_nodes, tf.convert_to_tensor(self.target_traj0_ten[0:num_nodes]),\
                                 tf.convert_to_tensor(pred_path), euc_loss0, # euc_loss0, fde0
                                 fde0, num_end_targets, euc_min, euc_idx]
            
            # print((params[0].graph.__repr__()) + '\n')
            
            euc_loss, fde_loss = evaluate(sess, graph_t, params, i, i0, adj_mat_vec)

            fde_loss = tf.reduce_sum(tf.linalg.norm(fde_loss, axis=2), axis=1) / batch_len
                                     # tf.reduce_min(, axis=0)

            euc_loss = tf.reduce_sum(euc_loss, axis=2)
            
            tmp = tf.linalg.norm(euc_loss, axis=2)
            euc_loss = tf.reduce_sum(tmp, axis=1)/(batch_len * 12)
            # euc_loss = tf.reduce_min(tmp)
            # idx = tf.argmin(tmp)
            
            # TODO pick minimum then optimize
            return euc_loss, fde_loss

     def h_to_a(self, num_nodes_sc, h_comp, w):
        # TODO construct kernel from random walk theory
        # TODO random walk is fast but least accurate model among graph completion algos
        # TODO check the literature for online nmf (OMF)
        # transform graph to kernel to parameterize
        # fNRI factorization of edges using the softmax
        #  make it variational, make n_proposals projections to generate n_proposals different permutations of adjacency
        # adj will all be ones, assuming fully connected graph at the init
        # use nmf to sparsify the adj, by making more plausible connections and less density graph.
        # TODO: restore all the adj_mat versions and select the best adjacency matrix for this pedestrian
        #  based on the least errors generated then run optimizer upon that.
        # adj = np.ones(shape=(h_comp.shape[0], h_comp.shape[0]))
        adj = np.ones(shape=(num_nodes_sc, num_nodes_sc), dtype=np.float64)
        adj_mat_vec = np.zeros(shape=(self.n_proposals, adj.shape[0], adj.shape[1]), dtype=np.float64)
        h_comp = np.array(h_comp, dtype=np.float64)
        w = np.array(w, dtype=np.float64)
        
        # print('Adjacency matrix length:{}'.format(num_nodes_sc))
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        # try:
        # while not coord.should_stop():
        # sess.run([hidden_state,attn], options=tf.RunOptions(timeout_in_ms=300))
        # except Exception as e:
        #     coord.request_stop(e)
        # finally:
        #     coord.request_stop()
        #     coord.join(threads)
        w = np.pad(w , [[1,1],[1,1]], mode='minimum')
        for k in range(self.n_proposals):
            # TODO see if map_fn applicable for online nmf
            w, h, n_iter = sk_dec.non_negative_factorization(X=adj, H=w, W=h_comp, init='custom',
                                                             n_components=adj.shape[0])
            adj_mat = np.matmul(w, h)
            adj_mat_vec[k] = adj_mat
        # edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
        # prob = my_softmax(logits, -1)
        # loss_kl = kl_categorical_uniform(prob, args.num_atoms, edge_types)
        return adj_mat_vec

    # Through gated neighborhood network (neighborhood encoders & random walker)

     # def select_best_rlns(self,n_adj, g):
     #
     #    #  map h onto adj_mat using h_to_a() function
     #    # make NMF on this adj_mat
     #    # infer relationships from the kernel (kernel output by random walker algorithm)
     #    # need to be created once at the init of master network.
     #    prob_mat = nn.sigmoid(n_adj)
     #    return prob_mat

     def eval_rln_ngh(self,adj_mat, combined_ngh):
        # This is the same mechanism used for choosing best ngh as (SGTV, 2019)
        # evaluate importance of relations to form the hybrid neighborhood(social(temporal) + static(spatial))
        # prob_mat = nn.Sigmoid(adj_mat)
        prob_mat = nn.softmax(adj_mat)

        return prob_mat

