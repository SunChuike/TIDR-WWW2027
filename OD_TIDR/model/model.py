#!/usr/bin/env python
# -*- coding:utf8 -*-
import sys
import os

cur_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.abspath(os.path.join(cur_path, '..')))
sys.path.append(os.path.abspath(os.path.join(cur_path, '../..')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util
from tensorflow.contrib.layers.python.layers.feature_column_ops import _input_from_feature_columns

import global_var as gl
from tensorflow.python.ops import metrics
from prada_interface.algorithm import Algorithm
from model_util.fg import FgParser
from model_util.util import *
from model_util.attention import attention as atten_func
from optimizer.adagrad_decay import SearchAdagradDecay
from optimizer.adagrad import SearchAdagrad
from optimizer.gradient_decent import SearchGradientDecent
from optimizer.gradient import SearchGradient
from optimizer import optimizer_ops as myopt
from tensorflow.python.framework.errors_impl import OutOfRangeError, ResourceExhaustedError
from requests.exceptions import ConnectionError
from tensorflow.contrib import layers
from fc_home_rank.trc_ctr_model.deepmatch_base_model_1.my_graph_learn import MyGraphLearn
from optimizer.adagrad_decay_zz import SearchAdagradDecayZz

import numpy as np
from model_util import odps_io as myodps

optimizer_dict = {
    "AdagradDecay": lambda opt_conf, global_step: SearchAdagradDecay(opt_conf).get_optimizer(global_step),
}

class ActiveNet(Algorithm):

    def variable_scope(self, *args, **kwargs):
        kwargs['partitioner'] = partitioned_variables.min_max_variable_partitioner(
            max_partitions=self.config.get_job_config("ps_num"),
            min_slice_size=self.config.get_job_config("dnn_min_slice_size"))
        kwargs['reuse'] = tf.AUTO_REUSE
        return tf.variable_scope(*args, **kwargs)

    def init(self, context):
        self.context = context
        self.logger = self.context.get_logger()
        self.config = self.context.get_config()

        gl._init()
        gl.set_value('logger', self.logger)

        for (k, v) in self.config.get_all_algo_config().items():
            self.model_name = k
            self.algo_config = v
            self.opts_conf = v['optimizer']
            self.model_conf = v['modelx']
            self.metric_conf = v['metrics']

        self.parse_param()

        # 这里完成Worker的 MyGraphLearn 的初始化
        self._my_graph_learn = MyGraphLearn(self.context, self.config)
        if not self._my_graph_learn.init_graph_learn_server():
            logger.error("Init graph learn server failed")
            return False
        logger.info("Run graph learn init suc")

        self.graph = self._my_graph_learn.get_graph()

        self.neg_sampler = self.graph.negative_sampler(
            'i',
            expand_factor=self.neg_num,
            strategy='node_weight',
            conditional=False,
            unique=False,
            batch_share=False
        )

        if self.model_name is None:
            self.model_name = "ActiveNet"

        self.user_column_blocks = []

        if self.algo_config.get('user_columns') is not None:
            arr_blocks = self.algo_config.get('user_columns').split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.user_column_blocks.append(block)

        self.context_column_blocks = []
        if self.algo_config.get('context_columns') is not None:
            arr_blocks = self.algo_config.get('context_columns').split(';', -1)
            for block in arr_blocks:
                if len(block) <= 0: continue
                self.context_column_blocks.append(block)

        self.seq_column_blocks = []
        self.seq_column_len = {}

        if self.algo_config.get('seq_column_blocks') is not None:
            arr_blocks = self.algo_config.get('seq_column_blocks').split(';', -1)
            for block in arr_blocks:
                if block == "":
                    continue
                arr = block.split(':', -1)
                if len(arr[0]) > 0:
                    self.seq_column_blocks.append(arr[0])
                if len(arr[1]) > 0:
                    self.seq_column_len[arr[0]] = arr[1]

        self.logger.info('sequence column blocks: {}'.format(self.seq_column_blocks))

        # Define model variables collection
        self.self_atten_collections_dnn_hidden_layer = "{}_self_atten_dnn_hidden_layer".format(self.model_name)
        self.self_atten_collections_dnn_hidden_output = "{}_self_atten_dnn_hidden_output".format(self.model_name)
        self.mlp_collections_dnn_hidden_layer = "{}_mlp_dnn_hidden_layer".format(self.model_name)
        self.mlp_collections_dnn_hidden_output = "{}_mlp_dnn_hidden_output".format(self.model_name)
        self.potential_emb_collections_dnn_hidden_layer = "{}_potential_emb_dnn_hidden_layer".format(self.model_name)
        self.potential_emb_collections_dnn_hidden_output = "{}_potential_emb_dnn_hidden_output".format(self.model_name)
        self.user_net_collections_dnn_hidden_layer = "{}_user_net_dnn_hidden_layer".format(self.model_name)
        self.user_net_collections_dnn_hidden_output = "{}_user_net_dnn_hidden_output".format(self.model_name)
        self.item_net_collections_dnn_hidden_layer = "{}_item_net_dnn_hidden_layer".format(self.model_name)
        self.item_net_collections_dnn_hidden_output = "{}_item_net_dnn_hidden_output".format(self.model_name)

        self.layer_dict = {}
        self.sequence_layer_dict = {}

        self.metrics = {}
        self.sink = context.get_sink()
        self.fg = FgParser(self.config.get_fg_config())
        self.debug_tensor_collector = {}

        try:
            self.is_training = tf.get_default_graph().get_tensor_by_name("training:0")
        except KeyError:
            self.is_training = tf.placeholder(tf.bool, name="training")

    def parse_param(self):
        self.neg_num = self.model_conf['model_hyperparameter'].get('neg_num', 20)
        self.temperature = self.model_conf['model_hyperparameter'].get('temperature', 0.02)
        self.dnn_l2_reg = self.model_conf['model_hyperparameter'].get('dnn_l2_reg', 0.0)
        self.activation = self.model_conf['model_hyperparameter'].get('activation', 'lrelu')
        self.comp_cate_num = self.model_conf['model_hyperparameter'].get('comp_cate_num', 4)
        self.comp_cate_dim = self.model_conf['model_hyperparameter'].get('comp_cate_dim', 64)
        self.ctx_dim = self.model_conf['model_hyperparameter'].get('ctx_dim', 128)
        self.decouple_loss_weight = self.model_conf['model_hyperparameter'].get('decouple_loss_weight', 0.1)
        self.query_num = self.comp_cate_num + 2

    def build_graph(self, context, features, feature_columns, labels):

        self.features = features[self.model_name]
        self.feature_columns = feature_columns[self.model_name]
        self.logger.info("[LogAllFeature] %s" % self.features)
        self.logger.info("[LogAllFeatureColumn] %s" % self.feature_columns)

        self.set_global_step()
        self.inference(self.features, self.feature_columns)
        self.loss()
        self.optimizer()
        self.mark_output()
        self.summary()
        self.set_reset_op()

    def set_reset_op(self):
        self.reset_ops, self.localvar = self.reset_variables(collection_key=tf.GraphKeys.LOCAL_VARIABLES,
                                                             matchname='Metrics/local')

    def reset_variables(self, collection_key=tf.GraphKeys.LOCAL_VARIABLES, matchname='auc/', not_match=None):
        localv = tf.get_collection(collection_key)
        self.logger.info("##### local variables: {}".format(localv))
        localv = [x for x in localv if matchname in x.name]
        if not_match is not None:
            localv = [x for x in localv if not_match not in x.name]
        self.logger.info("##### match local variables: {}".format(localv))
        retvops = [tf.assign(x, array_ops.zeros(shape=x.get_shape(), dtype=x.dtype)) for x in localv]
        if len(retvops) == 0:
            return None, None
        retvops = tf.tuple(retvops)
        return retvops, localv

    def set_global_step(self):
        """Sets up the global step Tensor."""
        self.global_step = training_util.get_or_create_global_step()
        self.global_step_reset = tf.assign(self.global_step, 0)
        self.global_step_add = tf.assign_add(self.global_step, 1, use_locking=True)
        tf.summary.scalar('global_step/' + self.global_step.name, self.global_step)

    def inference(self, features, feature_columns):
        self.embedding_layer(features, feature_columns)
        self.sequence_layer()
        self.user_net()
        self.item_net()
        self.logits_layer()

    def get_neg_attrs(self, ids):
        def _get_neg_attrs_func(ids):
            neg_nodes = self.neg_sampler.get(ids=ids)
            # [batch, neg_num, count(string)]
            return neg_nodes.string_attrs

        return tf.py_func(func=_get_neg_attrs_func,
                          inp=[ids], Tout=[tf.string])

    def build_sequence(self, seq_column_blocks, seq_column_len, scope):
        features = self.features
        feature_columns = self.feature_columns
        sequence_layer_dict = {}
        if seq_column_blocks is None or len(seq_column_blocks) == 0:
            return

        if len(seq_column_blocks) > 0:
            for block_name in seq_column_blocks:
                logger.info("Debug seq_column_bolcks: {}".format(block_name))
                if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                    raise ValueError('block_name:(%s) not in feature_columns for seq' % block_name)
                seq_len = self.fg.get_seq_len_by_sequence_name(block_name)
                self.logger.info("seq_len: {}".format(seq_len))
                sequence_stack = _input_from_feature_columns(features,
                                                             feature_columns[block_name],
                                                             weight_collections=None,
                                                             trainable=True,
                                                             scope=scope,
                                                             output_rank=3,
                                                             default_name='sequence_input_from_feature_columns')
                sequence_stack = tf.reshape(sequence_stack, [-1, seq_len, sequence_stack.get_shape()[(-1)].value])
                sequence_2d = tf.reshape(sequence_stack, [-1, tf.shape(sequence_stack)[2]])

                if block_name in seq_column_len and seq_column_len[block_name] in self.layer_dict:
                    sequence_length = self.layer_dict[seq_column_len[block_name]]
                    sequence_mask = tf.sequence_mask(tf.reshape(sequence_length, [-1]), seq_len)
                    self.logger.info("sequence_mask.type: {}".format(sequence_mask.dtype))

                    sequence_stack = tf.reshape(
                        tf.where(tf.reshape(sequence_mask, [-1]), sequence_2d, tf.zeros_like(sequence_2d)),
                        tf.shape(sequence_stack))
                else:
                    sequence_stack = tf.reshape(sequence_2d, tf.shape(sequence_stack))
                sequence_layer_dict[block_name] = sequence_stack

        return sequence_layer_dict

    def embedding_layer(self, features, feature_columns):
        with self.variable_scope("Embedding_Layer") as scope:

            for block_name in (self.seq_column_len.values()):
                if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                    raise ValueError("block_name:(%s) not in feature_columns for embed" % block_name)
                self.logger.info("block_name:%s, len(feature_columns[block_name])=%d" %
                                 (block_name, len(feature_columns[block_name])))

                self.layer_dict[block_name] = layers.input_from_feature_columns(features,
                                                                                feature_columns=feature_columns[
                                                                                    block_name],
                                                                                scope=scope)

            self.sequence_layer_dict = self.build_sequence(self.seq_column_blocks, self.seq_column_len, scope)

            self.pos_emb = layers.input_from_feature_columns(features, feature_columns['target_columns'], scope=scope)
            self.pos_item_ids_str = tf.squeeze(tf.sparse_tensor_to_dense(features['emb_item_id'], default_value='0'))
            self.pos_item_ids = tf.string_to_number(self.pos_item_ids_str, out_type=tf.int64)
            self.logger.info("##### pos_emb: {}".format(self.pos_emb))

            self.trigger_emb = layers.input_from_feature_columns(features, feature_columns['trigger_columns'], scope=scope)
            self.trigger_ids_str = tf.squeeze(tf.sparse_tensor_to_dense(features['trigger_id'], default_value='0'))
            self.trigger_ids = tf.string_to_number(self.trigger_ids_str, out_type=tf.int64)
            self.logger.info("##### trigger_emb: {}".format(self.trigger_emb)) # (?, 128)

            profile_embs = []
            for block_name in self.user_column_blocks:
                if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                    self.logger.info("skip missing profile block: %s" % block_name)
                    continue
                profile_embs.append(layers.input_from_feature_columns(features, feature_columns[block_name], scope=scope))
            self.profile_emb = tf.concat(profile_embs, -1) if len(profile_embs) > 0 else None
            self.logger.info("##### profile_emb: {}".format(self.profile_emb))

            context_embs = []
            for block_name in self.context_column_blocks:
                if block_name not in feature_columns or len(feature_columns[block_name]) <= 0:
                    self.logger.info("skip missing context block: %s" % block_name)
                    continue
                context_embs.append(layers.input_from_feature_columns(features, feature_columns[block_name], scope=scope))
            self.context_emb = tf.concat(context_embs, -1) if len(context_embs) > 0 else None
            self.logger.info("##### context_emb: {}".format(self.context_emb))

            self.comp_cate_emb = None
            if 'comp_cate_seq_list' in feature_columns and len(feature_columns['comp_cate_seq_list']) > 0:
                comp_seq = _input_from_feature_columns(features, feature_columns['comp_cate_seq_list'],
                                                       weight_collections=None,
                                                       trainable=True,
                                                       scope=scope,
                                                       output_rank=3,
                                                       default_name='comp_cate_input_from_feature_columns')
                comp_dim = comp_seq.get_shape().as_list()[-1]
                self.comp_cate_emb = tf.reshape(comp_seq, [-1, self.comp_cate_num, comp_dim])  # [B, K, d_cat]
            self.logger.info("##### comp_cate_emb: {}".format(self.comp_cate_emb))

            # neg
            attr_names = self.config.get_job_config('attr_names')
            self.neg_features = {}
            # [batch, neg_num, count(string)]
            self.neg_attrs = self.get_neg_attrs(self.pos_item_ids)
            # [batch*neg_num, count(string)]
            self.neg_attrs = tf.reshape(self.neg_attrs, [-1, len(attr_names)])
            for i in range(len(attr_names)):
                self.neg_features[attr_names[i]] = self.neg_attrs[:, i]
            self.logger.info("##### neg_attrs: {}".format(self.neg_attrs))
            self.logger.info("##### neg_features: {}".format(self.neg_features))

            # [batch*neg_num, dim]
            self.neg_emb = layers.input_from_feature_columns(self.neg_features, feature_columns['target_columns'],scope=scope)
            # [batch*neg_num]
            self.neg_item_ids_str = self.neg_features['emb_item_id']
            self.neg_item_ids = tf.string_to_number(self.neg_item_ids_str, out_type=tf.int64)
            self.logger.info("##### neg_emb: {}".format(self.neg_emb))
            self.logger.info("##### neg_item_ids: {}".format(self.neg_item_ids))

    def dnn_potential_emb_space(self, dnn_input, dnn_hidden_units, name):
        if isinstance(dnn_hidden_units, int):
            dnn_hidden_units = [dnn_hidden_units]

        with arg_scope(model_arg_scope(weight_decay=self.dnn_l2_reg)):
            for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                with self.variable_scope(name_or_scope="{}_potential_hidden_layer_{}".format(name, layer_id)) as dnn_hidden_layer_scope:
                    dnn_input = layers.fully_connected(
                        dnn_input,
                        num_hidden_units,
                        getActivationFunctionOp(self.activation),
                        scope=dnn_hidden_layer_scope,
                        normalizer_fn=layers.layer_norm,
                        normalizer_params={
                            "begin_norm_axis": -1,
                            "begin_params_axis": -1
                        },
                        variables_collections=[self.potential_emb_collections_dnn_hidden_layer],
                        outputs_collections=[self.potential_emb_collections_dnn_hidden_output]
                    )
        return dnn_input

    def mlp(self, dnn_input, dnn_hidden_units, name):

        if isinstance(dnn_hidden_units, int):
            dnn_hidden_units = [dnn_hidden_units]

        with arg_scope(model_arg_scope(weight_decay=self.dnn_l2_reg)):
            for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                with self.variable_scope(
                        name_or_scope="{}_mlp_hidden_layer_{}".format(name, layer_id)) as dnn_hidden_layer_scope:
                    dnn_input = layers.fully_connected(
                        dnn_input,
                        num_hidden_units,
                        getActivationFunctionOp(self.activation),
                        scope=dnn_hidden_layer_scope,
                        normalizer_fn=layers.layer_norm,
                        normalizer_params={
                            "begin_norm_axis": -1,
                            "begin_params_axis": -1
                        },
                        variables_collections=[self.mlp_collections_dnn_hidden_layer],
                        outputs_collections=[self.mlp_collections_dnn_hidden_output]
                    )
        return dnn_input

    def concat_available(self, tensors):
        avail = [t for t in tensors if t is not None]
        return tf.concat(avail, -1) if len(avail) > 1 else avail[0]

    def build_target_attention(self, queries, name):
        logger.info("Debug!! Start Build Target Attention: {}".format(name))
        with arg_scope(model_arg_scope(weight_decay=self.model_conf['model_hyperparameter']['atten_param']['attention_l2_reg'])):
            with self.variable_scope("build_target_attention_{}".format(name)) as scope:
                item_vec, stt_vec = atten_func(
                    query_masks=None,
                    key_masks=self.seq_mask,
                    queries=queries,
                    keys=self.seq_raw_input,
                    values=None,
                    num_units=self.model_conf['model_hyperparameter']['atten_param']['dill_sa_num_units'],
                    num_output_units=self.model_conf['model_hyperparameter']['atten_param']['dill_sa_num_output_units'],
                    scope=name + "_target_attention",
                    atten_mode=self.model_conf['model_hyperparameter']['atten_param']['atten_mode'],
                    reuse=tf.AUTO_REUSE,
                    variables_collections=[self.self_atten_collections_dnn_hidden_layer],
                    outputs_collections=[self.self_atten_collections_dnn_hidden_output],
                    num_heads=self.model_conf['model_hyperparameter']['atten_param']['num_heads'],
                    residual_connection=False,
                    attention_normalize=self.model_conf['model_hyperparameter']['atten_param'].get('attention_normalize',True),
                    use_atten_linear_project=self.model_conf['model_hyperparameter']['atten_param'].get('use_atten_linear_project', True))
                logger.info("Debug!! name is {}, item_vec is {}, stt_vec is {}".format(name, item_vec, stt_vec))
        logger.info("Debug!! Finsh Build Target Attention: {}".format(name))
        return item_vec

    def attention_pool(self, sequence, mask, name):
        with self.variable_scope("attention_pool_{}".format(name)) as scope:
            score = layers.fully_connected(sequence, 1, None, scope='pool_score',
                                           normalizer_fn=None, normalizer_params=None)
            score = tf.squeeze(score, -1)
            padding = (1.0 - tf.cast(mask, tf.float32)) * -1e9
            weight = tf.nn.softmax(score + padding, axis=1)
            return tf.reduce_sum(sequence * tf.expand_dims(weight, -1), axis=1)

    def sequence_layer(self):
        with self.variable_scope("Sequence_Layer") as scope:
            self.seq_raw_input = self.sequence_layer_dict['all_clk_seq_list']
            self.logger.info("##### seq_raw_input: {}".format(self.seq_raw_input))

            self.sequence_length_1d = tf.reshape(self.layer_dict[self.seq_column_len['all_clk_seq_list']], [-1])
            self.seq_mask = tf.sequence_mask(self.sequence_length_1d, tf.shape(self.seq_raw_input)[1])
            self.logger.info("##### sequence_length_1d: {}".format(self.sequence_length_1d))

            self.seq_core_dim = self.trigger_emb.get_shape().as_list()[-1]
            self.seq_core_input = self.seq_raw_input[:, :, :self.seq_core_dim]
            self.logger.info("##### seq_core_input: {}".format(self.seq_core_input))


    def user_net(self):
        with self.variable_scope(name_or_scope="user_net") as scope:
            dnn_hidden_units = self.model_conf['model_hyperparameter']['user_net_dnn_hidden_units']

            ctx_input = self.concat_available([self.trigger_emb, self.profile_emb, self.context_emb])
            self.h_ctx = self.mlp(ctx_input, [self.ctx_dim], 'ctx_mlp')

            if self.comp_cate_emb is not None:
                comp_cate = self.comp_cate_emb
            else:
                comp_cate_probe = tf.get_variable('comp_cate_probe',
                                                  shape=[self.comp_cate_num, self.comp_cate_dim],
                                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
                comp_cate = tf.tile(tf.expand_dims(comp_cate_probe, 0), [tf.shape(self.h_ctx)[0], 1, 1])
            d_cat = comp_cate.get_shape().as_list()[-1]

            h_ctx_tiled = tf.tile(tf.expand_dims(self.h_ctx, 1), [1, self.comp_cate_num, 1])
            film_input = tf.concat([h_ctx_tiled, comp_cate], -1)
            with arg_scope(model_arg_scope(weight_decay=self.dnn_l2_reg)):
                with self.variable_scope(name_or_scope="context_film") as film_scope:
                    gamma = layers.fully_connected(film_input, d_cat, None, scope='film_gamma',
                                                   normalizer_fn=None, normalizer_params=None)
                    beta = layers.fully_connected(film_input, d_cat, None, scope='film_beta',
                                                  normalizer_fn=None, normalizer_params=None)
            self.comp_cate_modulated = layers.layer_norm(gamma * comp_cate + beta,
                                                         begin_norm_axis=-1,
                                                         begin_params_axis=-1)

            trigger_query = tf.expand_dims(self.trigger_emb, 1)
            s_homo = self.build_target_attention(trigger_query, 'homo')
            s_homo = tf.squeeze(s_homo, 1)
            homo_input = self.concat_available([s_homo, self.trigger_emb, self.profile_emb])
            u_homo = self.mlp(homo_input, dnn_hidden_units, 'homo_mlp')
            u_homo = tf.nn.l2_normalize(u_homo, dim=-1)  # [B, d]

            s_comp = self.build_target_attention(self.comp_cate_modulated, 'comp')
            comp_input = tf.concat([self.comp_cate_modulated, s_comp, h_ctx_tiled], -1)
            u_comp = self.mlp(comp_input, dnn_hidden_units, 'comp_mlp')
            u_comp = tf.nn.l2_normalize(u_comp, dim=-1)

            tri_exp = tf.expand_dims(self.trigger_emb, 1)
            proj_coef = tf.reduce_sum(self.seq_core_input * tri_exp, -1, keepdims=True) / (
                    tf.reduce_sum(tri_exp * tri_exp, -1, keepdims=True) + 1e-9)
            seq_res = self.seq_core_input - proj_coef * tri_exp
            s_res = self.attention_pool(seq_res, self.seq_mask, 'res')
            res_input = self.concat_available([s_res, self.profile_emb])
            u_res = self.mlp(res_input, dnn_hidden_units, 'res_mlp')
            u_res = tf.nn.l2_normalize(u_res, dim=-1)

            self.user_vec = tf.concat([tf.expand_dims(u_homo, 1), u_comp, tf.expand_dims(u_res, 1)], axis=1)
            self.user_vec = tf.identity(self.user_vec, name='user_vec')
            self.logger.info("##### user_vec shape: {}".format(self.user_vec))

    def item_net(self):
        with self.variable_scope(name_or_scope="item_net") as scope:

            dnn_hidden_units = self.model_conf['model_hyperparameter']['item_net_dnn_hidden_units']

            dnn_output_units = self.pos_emb.get_shape().as_list()[-1]
            self.pos_item_vec = self.dnn_potential_emb_space(self.pos_emb, dnn_output_units, 'item')
            self.pos_item_vec = tf.concat([self.pos_emb, self.pos_item_vec], -1)
            self.neg_item_vec = self.dnn_potential_emb_space(self.neg_emb, dnn_output_units, 'item')
            self.neg_item_vec = tf.concat([self.neg_emb, self.neg_item_vec], -1)

            with arg_scope(model_arg_scope(weight_decay=0.0)):
                for layer_id, num_hidden_units in enumerate(dnn_hidden_units):
                    with self.variable_scope(
                            name_or_scope="item_hidden_layer_{}".format(layer_id)) as dnn_hidden_layer_scope:
                        self.pos_item_vec = layers.fully_connected(
                            self.pos_item_vec,
                            num_hidden_units,
                            getActivationFunctionOp(self.activation),
                            scope=dnn_hidden_layer_scope,
                            variables_collections=[self.item_net_collections_dnn_hidden_layer],
                            outputs_collections=[self.item_net_collections_dnn_hidden_output],
                            normalizer_fn=None,
                            normalizer_params=None)

                        self.neg_item_vec = layers.fully_connected(
                            self.neg_item_vec,
                            num_hidden_units,
                            getActivationFunctionOp(self.activation),
                            scope=dnn_hidden_layer_scope,
                            variables_collections=[self.item_net_collections_dnn_hidden_layer],
                            outputs_collections=[self.item_net_collections_dnn_hidden_output],
                            normalizer_fn=None,
                            normalizer_params=None)

            self.pos_item_vec = tf.nn.l2_normalize(self.pos_item_vec, dim=-1)
            self.neg_item_vec = tf.nn.l2_normalize(self.neg_item_vec, dim=-1)
            self.pos_item_vec = tf.identity(self.pos_item_vec, name='pos_item_vec')
            self.neg_item_vec = tf.identity(self.neg_item_vec, name='neg_item_vec')
            self.logger.info("##### pos_item_vec shape: {}".format(self.pos_item_vec)) # (?, 128)
            self.logger.info("##### neg_item_vec shape: {}".format(self.neg_item_vec)) # (?, 128)

    def logits_layer(self):
        with self.variable_scope("logits_layer"):
            B = tf.shape(self.user_vec)[0]
            D = self.user_vec.get_shape().as_list()[-1]

            pos_sim = tf.reduce_sum(self.user_vec * tf.expand_dims(self.pos_item_vec, 1), axis=-1)
            self.logger.info("##### pos_sim: {}".format(pos_sim))

            self.q_star = tf.argmax(pos_sim, axis=-1, output_type=tf.int32)
            slot_idx = tf.stack([tf.range(B), self.q_star], axis=1)

            user_expanded = tf.expand_dims(self.user_vec, axis=2)
            pos_expanded = tf.expand_dims(tf.expand_dims(self.pos_item_vec, axis=0), axis=1)
            sim = tf.reduce_sum(user_expanded * pos_expanded, axis=-1)
            self.inbatch_pos_logits = tf.gather_nd(sim, slot_idx) / self.temperature

            self.neg_logits = tf.matmul(
                tf.reshape(self.user_vec, [B * self.query_num, D]),
                self.neg_item_vec,
                transpose_b=True
            )
            self.neg_logits = tf.reshape(self.neg_logits, [B, self.query_num, -1])
            self.neg_logits = tf.gather_nd(self.neg_logits, slot_idx) / self.temperature

            self.pos_logits = tf.expand_dims(tf.gather_nd(pos_sim, slot_idx), -1) / self.temperature
            self.logger.info("##### pos_logits: {}".format(self.pos_logits))

    def reg_loss_f(self):
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_loss = tf.reduce_sum(reg_losses)

    def loss(self):
        with tf.name_scope("Loss_Op"):
            self.reg_loss_f()

            B = tf.shape(self.inbatch_pos_logits)[0]

            full_logits = tf.concat([self.inbatch_pos_logits, self.neg_logits], axis=-1)

            pos_labels = tf.eye(B)
            neg_labels = tf.zeros_like(self.neg_logits)
            full_labels = tf.concat([pos_labels, neg_labels], axis=-1)

            self.sampled_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=full_labels,
                logits=full_logits
            ))

            gram = tf.matmul(self.user_vec, self.user_vec, transpose_b=True)
            gram_identity = tf.eye(self.query_num)
            self.decouple_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(gram - gram_identity), axis=[1, 2]))

            self.loss_op = self.sampled_loss + self.decouple_loss_weight * self.decouple_loss + self.reg_loss

    def get_optimizer(self, opt_name, opt_conf, global_step):
        optimizer = None
        for name in optimizer_dict:
            if opt_name == name:
                optimizer = optimizer_dict[name](opt_conf, global_step)
                break

        return optimizer

    def optimizer(self):
        with self.variable_scope("Optimize"):

            global_opt_name = None
            global_optimizer = None
            global_opt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=None)

            if len(global_opt_vars) == 0:
                raise ValueError("no trainable variables")

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            train_ops = []
            for opt_name, opt_conf in self.opts_conf.items():
                optimizer = self.get_optimizer(opt_name, opt_conf, self.global_step)
                global_opt_name = opt_name
                global_optimizer = optimizer

            if global_opt_name is not None:
                train_op, self.out_gradient_norm, self.out_var_norm = myopt.optimize_loss(
                    loss=self.loss_op,
                    global_step=self.global_step,
                    learning_rate=self.opts_conf[global_opt_name].get("learning_rate", 0.01),
                    optimizer=global_optimizer,
                    clip_gradients=self.opts_conf[global_opt_name].get('clip_gradients', 5.0),
                    variables=global_opt_vars,
                    increment_global_step=False,
                    summaries=myopt.OPTIMIZER_SUMMARIES,
                )
                train_ops.append(train_op)

            with tf.control_dependencies(update_ops):
                train_op_vec = control_flow_ops.group(*train_ops)
                with ops.control_dependencies([train_op_vec]):
                    with ops.colocate_with(self.global_step):
                        self.train_ops = state_ops.assign_add(self.global_step, 1).op

    def predictions(self, logits):
        pass

    def mark_output(self):
        with tf.name_scope("Mark_Output"):
            rank_predict = tf.identity(self.pos_logits, name="rank_predict")
            pos_item_vec = tf.identity(self.pos_item_vec, name="pos_item_vec")
            user_vec = tf.identity(self.user_vec, name="user_vec")

            self.logger.info("##### rank_predict: {}".format(rank_predict))

    def summary(self):
        with tf.name_scope("{}_Metrics".format(self.model_name)):
            self.metrics['scalar/loss_mean'] = self.loss_op
            self.metrics['scalar/reg_loss'] = self.reg_loss
            self.metrics['scalar/sampled_loss'] = self.sampled_loss
            self.metrics['scalar/decouple_loss'] = self.decouple_loss

            # eval
            tf_conf = json.loads(os.environ['TF_CONFIG'])
            self.logger.info(('##### tf_conf:{}'.format(tf_conf)))
            task_index = tf_conf["task"]["index"]

            if task_index <= 1:

                top_50_indices_list = []
                all_interest_logits = []

                item_vec = tf.concat([self.pos_item_vec, self.neg_item_vec], 0)
                item_ids = tf.concat([self.pos_item_ids, self.neg_item_ids], 0)
                # [batch]
                labels = tf.cast(tf.range(tf.shape(self.pos_item_vec)[0]), tf.int64)

                B = tf.shape(labels)[0]
                M = tf.shape(item_vec)[0]

                recall_ks = [1, 3, 5, 10, 20, 30, 40, 50]

                top_k_indices_dict = {k: [] for k in recall_ks}

                for i in range(self.query_num):
                    # [batch, batch+batch*neg_num]
                    interest_logits = tf.matmul(self.user_vec[:, i, :], item_vec, transpose_b=True)
                    all_interest_logits.append(interest_logits)

                    # 只需计算一次 top-50，其他 k 从其中切片
                    _, top_50_indices = tf.nn.top_k(interest_logits, k=50)
                    top_50_indices_list.append(top_50_indices)

                    for k in recall_ks:
                        top_k_indices_dict[k].append(top_50_indices[:, :k])

                    current_recall_50, update_recall_50 = metrics.recall_at_k(
                        labels=labels,
                        predictions=interest_logits,
                        k=50,
                        name='local_interest_{}_recall_50'.format(i)
                    )
                    with tf.control_dependencies([update_recall_50]):
                        current_recall_50 = tf.identity(current_recall_50)
                    self.metrics["scalar/interest_{}_recall_50".format(i)] = current_recall_50

                self.top_50_indices = tf.stack(top_50_indices_list, axis=1)
                self.top_50_item_ids = tf.gather(item_ids, self.top_50_indices)
                self.logger.info("##### top_50_indices: {}".format(self.top_50_indices))
                self.logger.info("##### top_50_item_ids: {}".format(self.top_50_item_ids))

                labels_all = tf.cast(tf.expand_dims(labels, 1), tf.int32)

                for k in recall_ks:
                    top_k_indices_all = tf.concat(top_k_indices_dict[k], axis=1)
                    is_match_tensor = tf.cast(tf.equal(top_k_indices_all, labels_all), tf.float32)
                    is_match_any = tf.reduce_max(is_match_tensor, axis=-1)
                    recall_k = tf.reduce_mean(is_match_any)
                    self.metrics["scalar/a_all_interest_recall_{}".format(k)] = recall_k

                aggregated_logits = tf.reduce_max(tf.stack(all_interest_logits, axis=1), axis=1)

                pos_scores = tf.linalg.diag_part(aggregated_logits[:, :B])
                pos_scores = tf.expand_dims(pos_scores, axis=1)

                greater = aggregated_logits > pos_scores

                per_sample_count = tf.reduce_sum(tf.cast(greater, tf.float32), axis=1)

                per_sample_total = tf.cast(M - 1, tf.float32)
                per_sample_ratio = per_sample_count / per_sample_total

                self.metrics["scalar/avg_num_neg_greater_per_sample"] = tf.reduce_mean(per_sample_count)
                self.metrics["scalar/avg_ratio_neg_greater_per_sample"] = tf.reduce_mean(per_sample_ratio)

        with tf.name_scope("Metrics_Scalar"):
            for key, metric in self.metrics.items():
                tf.summary.scalar(name=key, tensor=metric)

        with tf.name_scope('{}_Embedding_Summary'.format(self.model_name)):
            set_name = set()
            add_embed_layer_norm(self.pos_emb, self.feature_columns['target_columns'], omit=set_name)
            add_embed_layer_norm(self.neg_emb, self.feature_columns['target_columns'], omit=set_name)
            add_embed_layer_norm(self.trigger_emb, self.feature_columns['trigger_columns'], omit=set_name)
            for block_name, layer in self.layer_dict.items():
                add_embed_layer_norm(layer, self.feature_columns[block_name], omit=set_name)

        # variable
        with tf.name_scope("{}_Self_Atten_Summary".format(self.model_name)):
            add_norm2_summary(self.self_atten_collections_dnn_hidden_layer)
            add_dense_output_summary(self.self_atten_collections_dnn_hidden_output)
            add_weight_summary(self.self_atten_collections_dnn_hidden_layer)

        with tf.name_scope("{}_potential_emb_Summary".format(self.model_name)):
            add_norm2_summary(self.potential_emb_collections_dnn_hidden_layer)
            add_dense_output_summary(self.potential_emb_collections_dnn_hidden_output)
            add_weight_summary(self.potential_emb_collections_dnn_hidden_layer)

        with tf.name_scope("{}_User_Net_Summary".format(self.model_name)):
            add_norm2_summary(self.user_net_collections_dnn_hidden_layer)
            add_dense_output_summary(self.user_net_collections_dnn_hidden_output)
            add_weight_summary(self.user_net_collections_dnn_hidden_layer)

        with tf.name_scope("{}_Item_Net_Summary".format(self.model_name)):
            add_norm2_summary(self.item_net_collections_dnn_hidden_layer)
            add_dense_output_summary(self.item_net_collections_dnn_hidden_output)
            add_weight_summary(self.item_net_collections_dnn_hidden_layer)

        return self.metrics

    def run_predict(self, context, mon_session, task_index, thread_index):
        if int(thread_index) != 0:  # predict with one thread
            self.logger.info("Skip thread_ind==%s" % str(thread_index))
            return

        self.odps = myodps.resetOdpsTable(self.algo_config.get('table_name'), task_id=task_index,
                                          local_mode=False, odps_user=self.algo_config.get('odps_user'))
        self.tablewriter = myodps.getTableWriter(self.odps,
                                                 self.algo_config.get('table_name'),
                                                 task_id=task_index,
                                                 ds_output='eval_part',
                                                 local_mode=False)

        predict_step = self.algo_config.get('predict_max_step', 100)
        localcnt = 0
        id_feature_tensor = self.features["id"]
        try:
            id_feature_tensor = tf.sparse_tensor_to_dense(id_feature_tensor, default_value="")
            self.logger.info("#qlLog# self.id_feature_tensor")
        except:
            pass

        run_ops = [self.prediction, self.label, id_feature_tensor]

        debug_tensor_names = []
        for tensor_name, tensor in self.debug_tensor_collector.items():
            debug_tensor_names.append(tensor_name)
            run_ops.append(tensor)

        print('global_variables')
        for variable_name in tf.global_variables():
            print(variable_name)

        while True:
            localcnt += 1
            feed_dict = {'training:0': False}

            try:
                run_res = mon_session.run(run_ops, feed_dict=feed_dict)
                prob, y, qid = run_res[:3]
                records = []
                for i in range(len(prob)):
                    one_res = [str(prob[i][0]), str(y[i][0]), str(qid[i][0])]
                    ex = []
                    for tensor_name, tensor_value in zip(debug_tensor_names, run_res[3:]):
                        ex.append('{}={}'.format(tensor_name, str(tensor_value[i].tolist())))
                    if len(ex) > 0:
                        ex_str = '#'.join(ex)
                        one_res.append(ex_str)
                    else:
                        one_res.append('-')
                    records.append(one_res)

                self.tablewriter.write(task_index, records)
                self.logger.info(
                    'model_name=%s, size=%s,  step_left=%s' % (self.model_name, str(len(prob)), str(predict_step)))

                predict_step -= 1
                if predict_step < 1:
                    break
            except (ResourceExhaustedError, OutOfRangeError) as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                break
            except ConnectionError as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                self.logger.info("Reset table writer")
                self.odps = myodps.resetOdpsTable(self.algo_config.get('table_name'), task_id=task_index,
                                                  local_mode=False, odps_user=self.algo_config.get('odps_user'))
                self.tablewriter = myodps.getTableWriter(self.odps,
                                                         self.algo_config.get('table_name'),
                                                         task_id=task_index,
                                                         ds_output='eval_part',
                                                         local_mode=False)
            except Exception as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))

        try:
            if self.tablewriter is not None:
                self.tablewriter.close()
        except ConnectionError as e:
            self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
            self.logger.info("Reset table writer when close")
            self.odps = myodps.resetOdpsTable(self.algo_config.get('table_name'), task_id=task_index,
                                              local_mode=False, odps_user=self.algo_config.get('odps_user'))
            self.tablewriter = myodps.getTableWriter(self.odps,
                                                     self.algo_config.get('table_name'),
                                                     task_id=task_index,
                                                     ds_output='eval_part',
                                                     local_mode=False)
            if int(task_index) != 0:
                raise RuntimeError("Reset table writer when close")

            time.sleep(60)
            self.logger.info("Finish run_predict, sleep")

    def run_evaluate(self, context, mon_session, task_index, thread_index):
        localcnt = 0
        while True:
            localcnt += 1
            run_ops = [self.global_step_add, self.global_step, self.metrics, self.label, self.localvar]
            try:
                feed_dict = {'training:0': False}
                _, global_step, metrics, labels, flocalv = mon_session.run(
                    run_ops, feed_dict=feed_dict)
                if len(self.localvar) > 0:
                    index = np.array([0, -1])
                    self.logger.info(('localcnt:%s\t' % str(localcnt)) + '//'.join([x.name for x in self.localvar]))
                    self.logger.info(('localcnt:%s\t' % str(localcnt)) + '//'.join([str(x[index]) for x in flocalv]))

                auc, totalauc = metrics['scalar/auc'], metrics['scalar/total_auc']
                self.logger.info(
                    'Global_Step:{}, poslabel:{}, auc={}, totalauc={} thread={}'.format(
                        str(global_step),
                        str(labels.sum()),
                        str(auc),
                        str(totalauc),
                        str(thread_index)))
                newmark = np.max(flocalv[0][np.array([0, -1])])
                if newmark > self.metric_conf['auc_compute'].get('true_positives', 20000):
                    self.logger.info("positive_num now:{}".format(str(newmark)))
                    self.logger.info("auc_reset_step:{}".format(str(1000)))
                    self.logger.info('reset auc ops run')
                    index = np.array([0, -1])
                    flocalv = mon_session.run(self.reset_auc_ops, feed_dict=feed_dict)
                    self.logger.info(
                        ('localcnt:{}\t'.format(str(localcnt))) + '//'.join([x.name for x in self.localvar]))
                    self.logger.info(
                        ('localcnt:{}\t'.format(str(localcnt))) + '//'.join([str(x[index]) for x in flocalv]))

            except (ResourceExhaustedError, OutOfRangeError) as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                break  # release all
            except ConnectionError as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
            except Exception as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))

    def run_train(self, context, mon_session, task_index, thread_index):
        localcnt = 0
        while True:
            localcnt += 1
            self.id = self.features["id"]
            run_ops = [self.global_step, self.loss_op, self.metrics, self.id]
            try:
                if task_index == 0:
                    feed_dict = {'training:0': False}
                    run_ops.extend(
                        [self.top_50_item_ids, self.trigger_ids, self.pos_item_ids, self.neg_item_ids,self.inbatch_pos_logits,self.neg_logits])
                    global_step, loss, metrics, id, top_50_item_ids, trigger_ids, pos_item_ids, neg_item_ids, inbatch_pos_logits, neg_logits \
                        = mon_session.run(run_ops, feed_dict=feed_dict)
                    self.logger.info(
                        'Global_Step:{}, loss={}, metrics={}, id={}, '
                        'top_50_item_ids={}, trigger_ids={}, pos_item_ids={}, neg_item_ids={}, inbatch_pos_logits={}, neg_logits={}'.format(
                            global_step,
                            loss,
                            metrics,
                            id,
                            ','.join(map(str, top_50_item_ids[0])),
                            trigger_ids,
                            pos_item_ids,
                            neg_item_ids,
                            inbatch_pos_logits,
                            neg_logits,
                        ))
                else:
                    feed_dict = {'training:0': True}
                    run_ops.append(self.train_ops)
                    global_step, loss, metrics, id, _ = mon_session.run(run_ops, feed_dict=feed_dict)

                    self.logger.info(
                        'Global_Step:{}, loss={}, metrics={}, id={}'.format(
                            global_step,
                            loss,
                            metrics,
                            id[0],
                        ))

            except (ResourceExhaustedError, OutOfRangeError) as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
                break  # release all
            except ConnectionError as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
            except Exception as e:
                self.logger.info('Got exception run : %s | %s' % (e, traceback.format_exc()))
