import networkx as nx
from functools import reduce, partial
from jax import random, jit, value_and_grad, grad
import jax.numpy as jnp
import jax
import tensornetwork as tn
from utils import disentangled_nodes, ternary_converter


class IsometricTensorNet(nx.DiGraph):

    def build_tree(self,
                   tree_depth: int,
                   number_of_children: int,
                   node: str,
                   chi: int,
                   dim: int,
                   top_node: bool=True):
        """Build tensor tree.

        Args:
            tree_depth: depth of a tree
            number_of_children: number pf children of each node
            node: name of the strating node
            chi: bond dimension
            dim: dimension of a local hilbert space
            top_node: is it the top node of a tree?"""

        if tree_depth != 0:
            out_dims = []  # will be filled by output dimensions
            for i in range(number_of_children):  # adding children via loop
                self.add_node(node + str(i))  # + node
                # + edge
                self.add_edges_from([(node, node + str(i), {'connection': (i, number_of_children)})])

                # recursively add children to children of node
                # and extract output dimensions
                out_dims.append(self.build_tree(tree_depth-1,
                                                number_of_children,
                                                node + str(i),
                                                chi,
                                                dim,
                                                top_node=False))

            # total dimension of all output indices
            total_out_dim = reduce(lambda x, y: x * y, out_dims)
            if top_node:
                in_dim = 1
            else:
                in_dim = chi if total_out_dim > chi else total_out_dim

            # attributes of node's tensor
            self.nodes[node]['in_dims'] = [in_dim]
            self.nodes[node]['out_dims'] = out_dims
            self.nodes[node]['tensor_node'] = True
            return in_dim

        else:
            # setting leafs as dangling indices
            self.nodes[node]['in_dims'] = [dim]
            self.nodes[node]['tensor_node'] = False
            return dim

    def glue_edges(self,
                   edges,
                   new_node_name):
        """Glue several edges by a node.

        Args:
            edges: list of NetworkX edges
            new_node_name: str, name of a new node"""

        # adding new intermediate node (usually disentangler)
        self.add_node(new_node_name)
        # modification of edges (inserting new node)
        for i, edge in enumerate(edges):
            # convolved indices of tensors before modification
            out_index, in_index = self.edges[edge]['connection'] 
            # remove old edge
            self.remove_edge(*edge)
            # adding edges connecting new node with tn instead of the old edge
            self.add_edges_from([(edge[0], new_node_name, {'connection': (out_index, i + len(edges))})])
            self.add_edges_from([(new_node_name, edge[1], {'connection': (i, in_index)})])
        # dims of new node indices
        dims = list(map(lambda x: self.nodes[x]['in_dims'][0], self.successors(new_node_name)))
        # adding attr. of the new node
        self.nodes[new_node_name]['in_dims'] = dims
        self.nodes[new_node_name]['out_dims'] = dims
        self.nodes[new_node_name]['tensor_node'] = True

    def _generate_node_tensor(self,
                              name,
                              manifold,
                              key):
        # if node is not a dangling edge
        if self.nodes[name]['tensor_node']:
            # dimension of input and output indices
            in_dims = self.nodes[name]['in_dims']
            out_dims = self.nodes[name]['out_dims']
            # total dimension of input and output
            total_in_dims = reduce(lambda x, y: x * y, in_dims)
            total_out_dims = reduce(lambda x, y: x * y, out_dims)
            # generating a random tensor
            tensor = manifold.random(key, (total_out_dims, total_in_dims))
            # tensor = jnp.eye(total_out_dims, total_in_dims, dtype=jnp.complex64)
            # d_tensor = random.normal(key, (*tensor.shape, 2))
            # d_tensor = d_tensor[..., 0] + 1j * d_tensor[..., 1]
            # tensor = tensor + 0.01 * d_tensor
            # tensor, _ = jnp.linalg.qr(tensor)
            return tensor
        else:
            pass

    def generate_params(self,
                        manifold,
                        key):
        """Generate a dict of jnp.ndarrays that are parameters of
        a network.

        Args:
            manifold: stiefel manifold
            key: PRNGKey

        Returns:
            pytree of parameters."""

        # generate dict of tensors defining tensor net
        keys = random.split(key, len(self))
        return {name: self._generate_node_tensor(name, manifold, keys[i]) for i, name in enumerate(self) if self.nodes[name]['tensor_node'] == True}

    def get_tensor_network(self,
                           params):
        """Generate tn out of parameters.

        Args:
            params: pytree of parameters

        Returns:
            dict of tn nodes."""

        # defining tn nodes
        tn_nodes = {key: tn.Node(params[key].reshape(self.nodes[key]['out_dims'] + self.nodes[key]['in_dims']), key) for key in self if self.nodes[key]['tensor_node'] == True}
        # setting edges between tn nodes
        for node, tn_node in tn_nodes.items():
            for succ in self.successors(node):
                if self.nodes[succ]['tensor_node'] == True:
                    ind_out, ind_in = self.edges[(node, succ)]['connection']
                    tn.connect(tn_nodes[node][ind_out], tn_nodes[succ][ind_in], '{}->{}'.format(node, succ))
                else:
                    ind_out, _ = self.edges[(node, succ)]['connection']
                    tn_nodes[node][ind_out].set_name(succ)
        return tn_nodes

    def braket(self,
               params_bra,
               params_ket,
               gate,
               indices):
        """Calculates <psi1|gate|psi2>.

        Args:
            params_bra: pytree of jnp.ndarrays
            params_ket: pytree of jnp.ndarrays
            gate: jnp.ndarray representing a gate
            indices: list with edges identifyers

        Returns:
            value of <psi1|gate|psi2>."""

        gate_node = tn.Node(gate)
        params_bra = {key: val.conj() for key, val in params_bra.items()}
        psi_bra = self.get_tensor_network(params_bra)
        psi_ket = self.get_tensor_network(params_ket)
        for node_bra, node_ket in zip(psi_bra.values(), psi_ket.values()):
            for edge_bra, edge_ket in zip(tn.get_all_dangling([node_bra]), tn.get_all_dangling([node_ket])):
                if edge_bra.name in indices:
                    ind = indices.index(edge_bra.name)
                    gate_node[ind]^edge_bra
                    gate_node[ind + len(indices)]^edge_ket
                else:
                    edge_bra^edge_ket
        total_tn = list(psi_bra.values()) + [gate_node] + list(psi_ket.values())
        return tn.contractors.auto(total_tn).tensor

    def scalar_prod(self,
               params_bra,
               params_ket,
               indices):
        """Calculates <psi1|psi2>.

        Args:
            params_bra: pytree of jnp.ndarrays
            params_ket: pytree of jnp.ndarrays
            indices: list with edges identifyers

        Returns:
            value of <psi1|psi2>."""

        params_bra = {key: val.conj() for key, val in params_bra.items()}
        psi_bra = self.get_tensor_network(params_bra)
        psi_ket = self.get_tensor_network(params_ket)
        for node_bra, node_ket in zip(psi_bra.values(), psi_ket.values()):
            for edge_bra, edge_ket in zip(tn.get_all_dangling([node_bra]), tn.get_all_dangling([node_ket])):
                    edge_bra^edge_ket
        total_tn = list(psi_bra.values()) + list(psi_ket.values())
        return tn.contractors.auto(total_tn).tensor

    def precondition(self,
                     params,
                     indices):
        """Apply preconditioner to gradients.

        Args:
            params: pytree of parameters
            indices: list of indices where to apply a gate"""

        env = grad(lambda x: self.scalar_prod(params, x, indices), holomorphic=True)(params)
        def local_precond(e, p):
            return e.T @ p
        return {name: local_precond(env[name], params) for name, params in params.items()}

    def loss(self,
             params_bra,
             params_ket,
             gate,
             indices):
        """Returns value of the loss.

        Args:
            params_bra: pytree of jnp.ndarrays
            params_ket: pytree of jnp.ndarrays
            gate: jnp.ndarray representing a gate
            indices: list with edges identifyers

        Returns:
            value of the loss function."""

        return 1 - jnp.abs(self.braket(params_bra, params_ket, gate, indices)) ** 2


class TernaryMERA(IsometricTensorNet):

    def build_mera(self,
                   mera_depth,
                   chi,
                   dim):
        """Generates graph of MERA tensor network.

        Args:
            mera_depth: dept of the MERA tensor network
            chi: bond dimension
            dim: dimension of the local Hilbert space"""

        self.add_node('_')
        self.build_tree(mera_depth, 3, '_', chi, dim)
        for i in range(2, mera_depth + 1):
            for j, pair in enumerate(disentangled_nodes(i)):
                self.glue_edges(((list(self.predecessors(pair[0]))[0], pair[0]),
                                (list(self.predecessors(pair[1]))[0], pair[1])), '{}_{}'.format(i, j))

    def _partial_mera(self, nodes):
        set_of_nodes = set(nodes)
        for node in nodes:
            set_of_nodes = set.union(set_of_nodes, self._partial_mera(list(self.predecessors(node))))
        return set_of_nodes

    def partial_mera(self,
                     nodes):
        """Returns partial MERA for subset of indeices.

        Args:
            nodes: list of nodes

        Returns:
            partial mera."""

        return self.subgraph(self._partial_mera(nodes))

    @partial(jit, static_argnums=(0, 5, 6, 7, 8, 9))
    def train(self,
              params_bra,
              params_ket,
              state,
              gate,
              indices,
              iters,
              layers,
              opt,
              use_precond):
        """Apply gate to the MERA tensor network.

        Args:
            params_bra: pytree of jnp.ndarrays
            params_ket: pytree of jnp.ndarrays
            state: state of the optimizer
            gate: gate
            indices: list of indices where to apply a gate
            iters: number of optimization iterations
            layers: number of mera layers
            opt: qgoptax optimizer
            use_precond: boolean showing whether to use
                preconditioner or not

        Returns:
            tuple of new parameters, new optimizer state and final value of the loss function."""

        id_indices = list(map(lambda x: ternary_converter(x, layers), indices))
        sub_mera = self.partial_mera(id_indices)
        loss_fn = lambda x: sub_mera.loss(x, params_ket, gate, id_indices)
        loss_and_grad_fn = value_and_grad(loss_fn)
        if use_precond:
            precond_fn = lambda x: sub_mera.precondition(x, id_indices)
        def train_step(params_bra, state, loss):
            loss, grads = loss_and_grad_fn(params_bra)
            if use_precond:
                rho = precond_fn(params_bra)
                params_bra, state = opt.update(grads, state, params_bra, rho)
            else:
                params_bra, state = opt.update(grads, state, params_bra)
            return params_bra, state, loss
        return jax.lax.fori_loop(0, iters, lambda i, vars: train_step(*vars), (params_bra, state, jnp.array(0.)))

    def get_psi(self,
                params,
                layers):
        """Calcluates the state out of the mera (only for small meras, up to
        27 qubits).

        Args:
            params: pytree of parameters
            layers: number of MERA layers

        Returns:
            state as jnp.ndarray."""

        tensor_net = self.get_tensor_network(params)
        indices = list(range(0, 3 ** layers))
        indices = map(lambda x: ternary_converter(x, layers), indices)
        dangling_edges = tn.get_all_dangling(tensor_net.values())
        dangling_edges = {edge.name: edge for edge in dangling_edges}
        dangling_edges = [dangling_edges[ind] for ind in indices] + tn.get_all_dangling([tensor_net['_']])
        return tn.contractors.auto(tensor_net.values(), dangling_edges).tensor

    def set_to_product_state(self,
                             params,
                             state,
                             list_of_states,
                             iters,
                             layers,
                             opt):
        """Sets MERA tensor network into product state.

        Args:
            params: pytree of jnp.ndarrays
            state: state of the optimizer
            list_of_states: list with jnp.ndarray representing
                local states
            iters: number of optimization iterations
            layers: number of mera layers
            opt: qgoptax optimizer

        Returns:
            tuple of new parameters, new optimizer state and final value of the loss function."""

        sub_meras = map(lambda ind: self.partial_mera([ternary_converter(ind, layers)]), range(3 ** layers))
        gates = map(lambda x: jnp.tensordot(x, x.conj(), axes=0), list_of_states)
        id_indices = map(lambda ind: [ternary_converter(ind, layers)], range(3 ** layers))
        loss_fn = lambda x: reduce(lambda prev, vals: prev + vals[0].loss(x, x, vals[1], vals[2]), [jnp.array(0.)] + list(zip(sub_meras, gates, id_indices)))
        loss_and_grad_fn = value_and_grad(loss_fn)
        def train_step(params, state, loss):
            loss, grads = loss_and_grad_fn(params)
            params, state = opt.update(grads, state, params)
            return params, state, loss
        return jax.lax.fori_loop(0, iters, lambda i, vars: train_step(*vars), (params, state, jnp.array(0.)))
