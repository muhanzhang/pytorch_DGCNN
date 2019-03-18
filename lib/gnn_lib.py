import ctypes
import numpy as np
import os
import sys
import torch
import pdb

class _gnn_lib(object):

    def __init__(self, args):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.lib = ctypes.CDLL('%s/build/dll/libgnn.so' % dir_path)

        self.lib.GetGraphStruct.restype = ctypes.c_void_p
        self.lib.PrepareBatchGraph.restype = ctypes.c_int
        self.lib.PrepareSparseMatrices.restype = ctypes.c_int
        self.lib.NumEdgePairs.restype = ctypes.c_int

        if sys.version_info[0] > 2:
            args = [arg.encode() for arg in args]  # str -> bytes for each element in args
        arr = (ctypes.c_char_p * len(args))()
        arr[:] = args
        self.lib.Init(len(args), arr)

        self.batch_graph_handle = ctypes.c_void_p(self.lib.GetGraphStruct())

    def _prepare_graph(self, graph_list, is_directed=0):    
        edgepair_list = (ctypes.c_void_p * len(graph_list))()
        list_num_nodes = np.zeros((len(graph_list), ), dtype=np.int32)
        list_num_edges = np.zeros((len(graph_list), ), dtype=np.int32)        
        for i in range(len(graph_list)):
            if type(graph_list[i].edge_pairs) is ctypes.c_void_p:
                edgepair_list[i] = graph_list[i].edge_pairs
            elif type(graph_list[i].edge_pairs) is np.ndarray:
                edgepair_list[i] = ctypes.c_void_p(graph_list[i].edge_pairs.ctypes.data)
            else:
                raise NotImplementedError

            list_num_nodes[i] = graph_list[i].num_nodes
            list_num_edges[i] = graph_list[i].num_edges
        total_num_nodes = np.sum(list_num_nodes)
        total_num_edges = np.sum(list_num_edges)

        self.lib.PrepareBatchGraph(self.batch_graph_handle, 
                                len(graph_list), 
                                ctypes.c_void_p(list_num_nodes.ctypes.data),
                                ctypes.c_void_p(list_num_edges.ctypes.data),
                                ctypes.cast(edgepair_list, ctypes.c_void_p),
                                is_directed)

        return total_num_nodes, total_num_edges

    def PrepareSparseMatrices(self, graph_list, is_directed=0):
        assert not is_directed
        total_num_nodes, total_num_edges = self._prepare_graph(graph_list, is_directed)
        
        n2n_idxes = torch.LongTensor(2, total_num_edges * 2)
        n2n_vals = torch.FloatTensor(total_num_edges * 2)

        e2n_idxes = torch.LongTensor(2, total_num_edges * 2)
        e2n_vals = torch.FloatTensor(total_num_edges * 2)

        subg_idxes = torch.LongTensor(2, total_num_nodes)
        subg_vals = torch.FloatTensor(total_num_nodes)

        idx_list = (ctypes.c_void_p * 3)()
        idx_list[0] = n2n_idxes.numpy().ctypes.data
        idx_list[1] = e2n_idxes.numpy().ctypes.data
        idx_list[2] = subg_idxes.numpy().ctypes.data

        val_list = (ctypes.c_void_p * 3)()
        val_list[0] = n2n_vals.numpy().ctypes.data
        val_list[1] = e2n_vals.numpy().ctypes.data
        val_list[2] = subg_vals.numpy().ctypes.data

        self.lib.PrepareSparseMatrices(self.batch_graph_handle,
                                ctypes.cast(idx_list, ctypes.c_void_p),
                                ctypes.cast(val_list, ctypes.c_void_p))

        n2n_sp = torch.sparse.FloatTensor(n2n_idxes, n2n_vals, torch.Size([total_num_nodes, total_num_nodes]))
        e2n_sp = torch.sparse.FloatTensor(e2n_idxes, e2n_vals, torch.Size([total_num_nodes, total_num_edges * 2]))
        subg_sp = torch.sparse.FloatTensor(subg_idxes, subg_vals, torch.Size([len(graph_list), total_num_nodes]))
        return n2n_sp, e2n_sp, subg_sp

dll_path = '%s/build/dll/libgnn.so' % os.path.dirname(os.path.realpath(__file__))
if os.path.exists(dll_path):
    GNNLIB = _gnn_lib(sys.argv)
else:
    GNNLIB = None

