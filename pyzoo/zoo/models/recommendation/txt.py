#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from gluonnlp.model.transformer import TransformerEncoder, TransformerEncoderCell
from mxnet.gluon.block import HybridBlock


class MeanMaxPooling(HybridBlock):
    def __init__(self, axis=1, dropout=0.0, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self.axis = axis
        self.dropout = dropout

    def hybrid_forward(self, F, inputs):
        mean_out = F.mean(data=inputs, axis=self.axis)
        max_out = F.max(data=inputs, axis=self.axis)
        outputs = F.concat(mean_out, max_out, dim=1)
        if self.dropout:
            outputs = F.Dropout(data=outputs, p=self.dropout)
        outputs = F.LayerNorm(data=outputs)
        return outputs


class SequenceTransformer(HybridBlock):
    def __init__(self, num_items, item_embed, item_hidden_size, item_max_length, item_num_heads,
                 item_num_layers, item_transformer_dropout, item_pooling_dropout, cross_size,
                 prefix=None,
                 params=None):
        super().__init__(prefix=prefix, params=params)
        self.num_items = num_items
        self.item_embed = item_embed
        self.cross_size = cross_size
        with self.name_scope():
            self.item_pooling_dp = MeanMaxPooling(dropout=item_pooling_dropout)
            self.item_encoder = TransformerEncoder(units=item_embed, hidden_size=item_hidden_size,
                                                   num_heads=item_num_heads,
                                                   num_layers=item_num_layers,
                                                   max_length=item_max_length,
                                                   dropout=item_transformer_dropout)

    def hybrid_forward(self, F, input_item, item_valid_length=None):
        item_embed_out = F.Embedding(data=input_item, input_dim=self.num_items,
                                     output_dim=self.item_embed)
        item_encoding, item_att = self.item_encoder.hybrid_forward(F, inputs=item_embed_out,
                                                                   valid_length=item_valid_length)
        item_out = self.item_pooling_dp.hybrid_forward(F, inputs=item_encoding)
        item_out = F.FullyConnected(data=item_out, num_hidden=self.cross_size)

        return item_out


class ContextTransformer(HybridBlock):
    def __init__(self, context_dims, context_embed, context_hidden_size,
                 context_num_heads, context_transformer_dropout, context_pooling_dropout,
                 cross_size, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self.context_dims = context_dims
        self.context_embed = context_embed
        self.cross_size = cross_size
        with self.name_scope():
            self.context_pooling_dp = MeanMaxPooling(dropout=context_pooling_dropout)
            self.context_encoder = TransformerEncoderCell(units=context_embed,
                                                          hidden_size=context_hidden_size,
                                                          num_heads=context_num_heads,
                                                          dropout=context_transformer_dropout
                                                          )

    def hybrid_forward(self, F, input_context_list):
        context_embed = [F.Embedding(data=input_context_list[i], input_dim=self.context_dims[i],
                                     output_dim=self.context_embed)
                         for i, context_dim in enumerate(self.context_dims)]
        context_input = []
        for i in context_embed:
            context_input.append(F.expand_dims(i, axis=1))
        context_embedding = F.concat(*context_input, dim=1)
        context_encoding, context_att = self.context_encoder. \
            hybrid_forward(F, inputs=context_embedding)
        context_out = self.context_pooling_dp.hybrid_forward(F, inputs=context_encoding)
        context_out = F.FullyConnected(data=context_out, num_hidden=self.cross_size)

        return context_out


class TxT(HybridBlock):
    def __init__(self, num_items, context_dims, item_embed=100, context_embed=100,
                 item_hidden_size=256, item_max_length=8, item_num_heads=4, item_num_layers=2,
                 item_transformer_dropout=0.0, item_pooling_dropout=0.1, context_hidden_size=256,
                 context_num_heads=2, context_transformer_dropout=0.0,
                 context_pooling_dropout=0.0, act_type="relu", cross_size=100,
                 prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self.num_items = num_items
        self.act_type = act_type
        with self.name_scope():
            self.sequence_transformer = SequenceTransformer(
                num_items=num_items, item_embed=item_embed,
                item_hidden_size=item_hidden_size,
                item_max_length=item_max_length,
                item_num_heads=item_num_heads,
                item_num_layers=item_num_layers,
                item_transformer_dropout=item_transformer_dropout,
                item_pooling_dropout=item_pooling_dropout,
                cross_size=cross_size,
                prefix=prefix, params=params
            )
            self.context_transformer = ContextTransformer(
                context_dims=context_dims,
                context_embed=context_embed,
                context_hidden_size=context_hidden_size,
                context_num_heads=context_num_heads,
                context_transformer_dropout=context_transformer_dropout,
                context_pooling_dropout=context_pooling_dropout,
                cross_size=cross_size,
                prefix=prefix, params=params
            )

    def hybrid_forward(self, F, input_item, input_context_list,
                       label, item_valid_length=None):
        item_outs = self.sequence_transformer.hybrid_forward(F, input_item=input_item,
                                                             item_valid_length=item_valid_length)
        context_outs = self.context_transformer.hybrid_forward(
            F, input_context_list=input_context_list
        )
        outs = F.broadcast_mul(item_outs, context_outs)
        outs = F.Activation(data=outs, act_type=self.act_type)
        outs = F.FullyConnected(data=outs, num_hidden=int(self.num_items))
        outs = F.SoftmaxOutput(data=outs, label=label)

        return outs
