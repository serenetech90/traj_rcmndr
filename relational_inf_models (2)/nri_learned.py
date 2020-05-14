import numpy as np
from tensorflow import nn
import tensorflow as tf
import sklearn.decomposition as sk_dec
from tensorflow.contrib import learn as skf
import pathos.multiprocessing as mp
from threading import Thread

@tf.contrib.eager.defun

# pool = mp.Pool(processes=mp.cpu_count())
def evaluate(sess, sgraph, params,i , itr):
        # nri_mp = nri_learned(args)
        def condition(pred_len, attn, hidden_state, num_nodes_sc, num_nodes,
                      target_traj, pred_path, euc_loss, fde, num_end_targets, i, _itr=0):
            # _, _, _, num_nodes, _, _, _, _, _, _ = _params
            return i < num_nodes_sc

        def inner_loop_fn(pred_len, attn, hidden_state, num_nodes_sc, num_nodes, target_traj, pred_path, euc_loss, fde, num_end_targets, i, itr):
            # sess = tf.InteractiveSession()
            try:
                # with graph.as_default():
                    pred_len, attn, hidden_state, num_nodes_sc, num_nodes, target_traj, pred_path, euc_loss, fde, num_end_targets = params
                    with tf.Session(graph=hidden_state.graph) as sess:
                        # hidden_state, attn = sess.run(fetches=[hidden_state,attn])
                        # TODO: restore all the adj_mat versions and select the best adjacency matrix for this pedestrian
                        #  based on the least errors generated then run optimizer upon that.
                        adj = np.ones(shape=(num_nodes_sc, num_nodes_sc))
                        adj_mat_vec = np.zeros(shape=(10, adj.shape[0], adj.shape[1]), dtype=np.float32)
                        # coord = tf.train.Coordinator()
                        # threads = tf.train.start_queue_runners(coord=coord)
                        # try:
                        # while not coord.should_stop():
                        hidden_state, attn = sess.run(fetches=[hidden_state,attn], options=tf.RunOptions(timeout_in_ms=300))
                        # hidden_state.eval(session=sess), attn.eval(session=sess)
                        # except Exception as e:
                        #     coord.request_stop(e)
                        # finally:
                        #     coord.request_stop()
                        #     coord.join(threads)

                        for k in range(10):
                            # TODO see if map_fn applicable for online nmf
                            w, h, n_iter = sk_dec.non_negative_factorization(X=adj, H=hidden_state[k], W=attn, init='custom',
                                                                                       n_components=adj.shape[0])
                            adj_mat = tf.matmul(w, h)
                            adj_mat_vec[k] = adj_mat

                        # def h_to_a_call():
                        # adj_mat = nri_mp.h_to_a(h_comp=hidden_state, w=attn,
                        #                          adj=tf.ones(shape=(num_nodes, num_nodes)))
                        hidden_state = np.matmul(attn, hidden_state)

                        # adj_mat = tf.matmul(tf.nn.softmax(hidden_state),
                        #                     tf.ones(shape=(hidden_state.shape[1], 1), dtype=tf.float64))
                        # # GG-NN 2016 A_nx2n , we use A_nxn; n is |G_v| cardinality of node v in Graph G.
                        hidden_state = adj_mat_vec * hidden_state

                        num_end_targets += 1
                        if target_traj[i].shape[0] < pred_len:
                            # krnl_mdl.pred_path_band.eval()
                            # loss = tf.reduce_sum(tf.square(y0 - y_out))
                            euc_loss.append(pred_path[i][0:target_traj[itr].shape[0]] - target_traj[itr])
                            fde.append(pred_path[i][target_traj[itr].shape[0] - 1] - target_traj[itr][
                                target_traj[i].shape[0] - 1])
                            # euc_loss = np.linalg.norm((pred_path[i][0:len(target_traj[i])] - target_traj[i]), ord=2)/len(target_traj)
                        else:
                            euc_loss.append(
                                pred_path[i][0:pred_len] - target_traj[itr][0:pred_len])
                            fde.append(pred_path[i][pred_len - 1] - target_traj[itr][
                                pred_len - 1])
            except KeyError:
                pass
                # i+=1

        # for i, itr in zip(range(1, num_nodes), iter(target_traj)):
        # cond = lambda i, num_nodes: i < num_nodes #and itr in range(len(target_traj))
        # body = tf.map_fn(fn=lambda params,i , itr: inner_loop_fn(params, i, itr),
        #                  elems=(params, i, itr), dtype=tf.float32, parallel_iterations=50)
        k = 0

        pred_len, attn, hidden_state, num_nodes_sc, num_nodes, target_traj, pred_path, euc_loss, fde, num_end_targets = params

        body = lambda pred_len, attn, hidden_state, num_nodes_sc, num_nodes,\
                                                    target_traj, pred_path, euc_loss, fde, num_end_targets, i, itr: \
                                                    inner_loop_fn(pred_len, attn, hidden_state, num_nodes_sc, num_nodes,
                                                    target_traj, pred_path, euc_loss, fde, num_end_targets, i, itr)

        tf.while_loop(cond=condition, body=body, loop_vars=[pred_len, attn, hidden_state, num_nodes_sc, num_nodes,
                                    target_traj, pred_path, euc_loss, fde, num_end_targets, k, itr],
                                    parallel_iterations=10)
        return euc_loss, fde

def extract_ten_dict(dict, ten=[]):
    # rows_lens = []
    # due to limitations in converting to ragged tensor from list of variable-length tensors,
    # there is no other way in TF except fixing the size.
    # problem in casting type list to Tensor
    # max_ten = tf.zeros(shape=(len(dict), 12, 2))
    for itr in dict:
        if dict[itr].shape[0] < 12: # max_ten[itr].shape[0]:
            dict[itr] = tf.concat((dict[itr], tf.zeros(shape=(abs(dict[itr].shape[0].value - 12), 2))), axis=0)
        # max_ten[itr] = tf.assign(max_ten[itr], (max_ten[itr] + dict[itr]))
        # rows_lens.append(int(dict[itr].shape[0]))

    return dict


class nri_learned():
     def __init__(self, args, sess):
        super(nri_learned, self).__init__()
        self.args = args
        self.sess = sess
        self.init = tf.random_normal(mean=0, stddev=1, seed=0, dtype=tf.float64, shape=[1, ])
        self.l2norm_vec = tf.Variable(initial_value=self.init, dtype=tf.float64, name='l2norm_vec')
        self.loss = tf.nn.l2_loss(t=self.l2norm_vec, name='loss')
        self.loss_optzr = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate).minimize(self.loss)

        # self.threaded_fn = Thread(target=self.h_to_a, daemon=True)
        # self.threaded_fn.start()

     def assess_rcmndr(self,sess, graph_t, num_nodes, euc_loss, fde, pred_path, attn, hidden_state, target_traj):
         # with tf.Session(graph=graph_t).as_default() as sess:
            target_traj0 = tf.feature_column.categorical_column_with_vocabulary_list(key='ID', vocabulary_list=target_traj)

            # Transform all variables to a hashable format (tf tensors) that can be iterable in the eager_execution mode
            # inside while_loop() call

            itr = tf.convert_to_tensor(target_traj0.vocabulary_list)
            target_traj0_ten = []
            target_traj0_ten = extract_ten_dict(target_traj, target_traj0_ten)
            # i0 = tf.constant(value=0)
            pred_len0 = tf.convert_to_tensor(value=[self.args.pred_len])
            i = tf.convert_to_tensor([tf.constant(value=0)])
            num_nodes0 = tf.convert_to_tensor([num_nodes])
            hidden_state0 = tf.convert_to_tensor(hidden_state)
            euc_loss0 = tf.convert_to_tensor(euc_loss)
            fde0 = tf.convert_to_tensor(fde)
            num_end_targets = tf.zeros(shape=(1,))

            params = [pred_len0, attn, hidden_state0,
                      num_nodes, num_nodes0, target_traj0_ten, pred_path, euc_loss0, fde0, num_end_targets]
            # params = np.array(params)
            euc_loss, fde_loss = evaluate(sess, graph_t, params, i, itr)
            fde_loss = tf.cumsum(fde_loss) / num_nodes
            euc_loss = tf.cumsum(euc_loss) / (num_nodes * 12)
            print('L2-Loss = ',euc_loss)
            print('FDE = ', fde_loss)
            # TODO pick minimum then optimize
            return euc_loss, fde_loss

     def h_to_a(self, h_comp, w, adj):
        # TODO construct kernel from random walk theory
        # TODO random walk is fast but least accurate model among graph completion algos
        # TODO check the literature for online nmf (OMF)
        # transform graph to kernel to parameterize
        # fNRI factorization of edges using the softmax
        #  make it variational, make 10 projections to generate 10 different permutations of adjacency
        # adj will all be ones, assuming fully connected graph at the init
        # use nmf to sparsify the adj, by making more plausible connections and less density graph.
        adj_mat_vec = tf.zeros(shape=(10, adj.shape[0], adj.shape[1]), dtype=tf.float32)
        for k in range(10):
            w, h, n_iter = sk_dec.non_negative_factorization(X=adj,H=h_comp[k], W=w, init='custom',
                                                                       n_components=adj.shape[0])
            adj_mat = tf.matmul(w, h)
            adj_mat_vec[k] = adj_mat

        # edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
        # prob = my_softmax(logits, -1)
        # loss_kl = kl_categorical_uniform(prob, args.num_atoms, edge_types)
        return adj_mat_vec

    # Through gated neighborhood network (neighborhood encoders & random walker)
     def select_best_rlns(self,n_adj, g):
        #  map h onto adj_mat using h_to_a() function
        # make NMF on this adj_mat
        # infer relationships from the kernel (kernel output by random walker algorithm)
        # need to be created once at the init of master network.
        # self.sig =
        prob_mat = nn.sigmoid(n_adj)
        return prob_mat

     def eval_rln_ngh(self,adj_mat, combined_ngh):

        # evaluate importance of relations to form the hybrid neighborhood(social(temporal) + static(spatial))
        # prob_mat = nn.Sigmoid(adj_mat)
        prob_mat = nn.softmax(adj_mat)

        return prob_mat

