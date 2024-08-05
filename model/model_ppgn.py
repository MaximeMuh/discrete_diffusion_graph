import torch
import torch.nn as nn
import torch.nn.functional as F

SLOPE = 0.01

class PowerfulLayer(nn.Module):
    def __init__(
        self,
        in_feat: int,
        out_feat: int,
        num_layers: int,
        activation=nn.LeakyReLU(negative_slope=SLOPE),
        spectral_norm=(lambda x: x),
    ):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.m1 = nn.Sequential(
            *[
                spectral_norm(nn.Linear(in_feat if i == 0 else out_feat, out_feat))
                if i % 2 == 0
                else activation
                for i in range(num_layers * 2 - 1)
            ]
        )
        self.m2 = nn.Sequential(
            *[
                spectral_norm(nn.Linear(in_feat if i == 0 else out_feat, out_feat))
                if i % 2 == 0
                else activation
                for i in range(num_layers * 2 - 1)
            ]
        )
        self.m4 = nn.Sequential(
            spectral_norm(nn.Linear(in_feat + out_feat, out_feat, bias=True))
        )

    def forward(self, x):
        """x: batch x N x N x in_feat"""

        norm = x.size(1) ** 0.5  # Normalization factor

        # Apply transformations and permutation
        out1 = self.m1(x).permute(0, 3, 1, 2)  # batch, out_feat, N, N
        out2 = self.m2(x).permute(0, 3, 1, 2)  # batch, out_feat, N, N

        # Matrix multiplication
        out = out1 @ out2
        out = out / norm

        # Concatenate with skip connection and apply final transformation
        out = torch.cat((out.permute(0, 2, 3, 1), x), dim=3)  # batch, N, N, out_feat
        out = self.m4(out)

        return out


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation=nn.LeakyReLU(negative_slope=SLOPE),
        spectral_norm=(lambda x: x),
    ):
        super().__init__()
        self.lin1 = nn.Sequential(
            spectral_norm(nn.Linear(in_features, out_features, bias=True))
        )
        self.lin2 = nn.Sequential(
            spectral_norm(nn.Linear(in_features, out_features, bias=False))
        )
        self.lin3 = nn.Sequential(
            spectral_norm(nn.Linear(out_features, out_features, bias=False))
        )
        self.activation = activation

    def forward(self, u):
        """u: (batch_size, num_nodes, num_nodes, in_features)"""
        # Compute the diagonal and trace
        n = u.size(1)
        diag = u.diagonal(dim1=1, dim2=2)
        trace = torch.sum(diag, dim=2)

        # Apply transformations
        out1 = self.lin1(trace / n)
        s = (torch.sum(u, dim=[1, 2]) - trace) / (n * (n - 1))
        out2 = self.lin2(s)
        out = out1 + out2
        out = out + self.lin3(self.activation(out))

        return out


class Powerful(nn.Module):
    def __init__(
        self,
        use_norm_layers: bool,
        name: str,
        channel_num_list,
        feature_nums: list,
        gnn_hidden_num_list: list,
        num_layers: int,
        input_features: int,
        hidden: int,
        hidden_final: int,
        dropout_p: 0.000001,
        simplified: bool,
        n_nodes: int,
        device,
        normalization: str = "none",
        cat_output: bool = True,
        adj_out: bool = False,
        output_features: int = 1,
        residual: bool = False,
        activation=nn.LeakyReLU(negative_slope=SLOPE),
        spectral_norm=(lambda x: x),
        project_first: bool = False,
        node_out: bool = False,
        noise_mlp: bool = False,
    ):
        super().__init__()
        self.cat_output = cat_output
        self.normalization = normalization
        layers_per_conv = 2
        self.layer_after_conv = not simplified
        self.dropout_p = dropout_p
        self.adj_out = adj_out
        self.residual = residual
        self.activation = activation
        self.project_first = project_first
        self.node_out = node_out
        self.output_features = output_features
        self.node_output_features = output_features
        self.noise_mlp = noise_mlp
        self.device = device

        # For noiselevel conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 4),
            nn.GELU(),
            nn.Linear(4, 1),
        )

        self.in_lin = nn.Sequential(spectral_norm(nn.Linear(input_features, hidden)))

        if self.cat_output:
            if self.project_first:
                self.layer_cat_lin = nn.Sequential(
                    spectral_norm(nn.Linear(hidden * (num_layers + 1), hidden))
                )
            else:
                self.layer_cat_lin = nn.Sequential(
                    spectral_norm(
                        nn.Linear(hidden * num_layers + input_features, hidden)
                    )
                )
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        for i in range(num_layers):
            self.convs.append(
                PowerfulLayer(
                    hidden,
                    hidden,
                    layers_per_conv,
                    activation=self.activation,
                    spectral_norm=spectral_norm,
                )
            )

        self.feature_extractors = nn.ModuleList([])
        for i in range(num_layers):
            if self.normalization == "layer":
                self.bns.append(
                    nn.LayerNorm([n_nodes, n_nodes, hidden], elementwise_affine=False)
                )
            elif self.normalization == "batch":
                self.bns.append(nn.BatchNorm2d(hidden))
            else:
                self.bns.append(None)
            self.feature_extractors.append(
                FeatureExtractor(
                    hidden,
                    hidden_final,
                    activation=self.activation,
                    spectral_norm=spectral_norm,
                )
            )
        if self.layer_after_conv:
            self.after_conv = nn.Sequential(
                spectral_norm(nn.Linear(hidden_final, hidden_final))
            )
        self.final_lin = nn.Sequential(
            spectral_norm(nn.Linear(hidden_final, output_features))
        )

        if self.node_out:
            if self.cat_output:
                if self.project_first:
                    self.layer_cat_lin_node = nn.Sequential(
                        spectral_norm(nn.Linear(hidden * (num_layers + 1), hidden))
                    )
                else:
                    self.layer_cat_lin_node = nn.Sequential(
                        spectral_norm(
                            nn.Linear(hidden * num_layers + input_features, hidden)
                        )
                    )
            if self.layer_after_conv:
                self.after_conv_node = nn.Sequential(
                    spectral_norm(nn.Linear(hidden_final, hidden_final))
                )
            self.final_lin_node = nn.Sequential(
                spectral_norm(nn.Linear(hidden_final, node_output_features))
            )

    def get_out_dim(self):
        return self.output_features

    def forward(self, node_features, A, noiselevel):
        out = self.forward_cat(A, node_features, noiselevel)
        return out

    def forward_cat(self, A, node_features, noiselevel):
        if len(A.shape) < 4:
            u = A[..., None]  # batch, N, N, 1
        else:
            u = A

        # Condition the noise level
        if self.noise_mlp:
            noiselevel = self.time_mlp(torch.full([1], noiselevel).to(self.device))
            noise_level_matrix = noiselevel.expand(u.size(0), u.size(1), u.size(3)).to(self.device)
            noise_level_matrix = torch.diag_embed(
                noise_level_matrix.transpose(-2, -1), dim1=1, dim2=2
            )
        else:
            noiselevel = torch.full([1], noiselevel).to(self.device)
            noise_level_matrix = noiselevel.expand(u.size(0), u.size(1), u.size(3)).to(self.device)
            noise_level_matrix = torch.diag_embed(
                noise_level_matrix.transpose(-2, -1), dim1=1, dim2=2
            )

        u = torch.cat([u, noise_level_matrix], dim=-1).to(self.device)

        if self.project_first:
            u = self.in_lin(u)
            out = [u]
        else:
            out = [u]
            u = self.in_lin(u)

        for conv, extractor, bn in zip(self.convs, self.feature_extractors, self.bns):
            u = conv(u) + (u if self.residual else 0)
            if self.normalization == "none":
                u = u
            elif self.normalization == "instance":
                u = u  # Instance normalization removed
            elif self.normalization == "layer":
                u = bn(u)
            elif self.normalization == "batch":
                u = bn(u.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            else:
                raise ValueError
            out.append(u)
        out = torch.cat(out, dim=-1)
        if self.node_out and self.adj_out:
            node_out = self.layer_cat_lin_node(
                out.diagonal(dim1=1, dim2=2).transpose(-2, -1)
            )
            if self.layer_after_conv:
                node_out = node_out + self.activation(self.after_conv_node(node_out))
            node_out = F.dropout(node_out, p=self.dropout_p, training=self.training)
            node_out = self.final_lin_node(node_out)
        out = self.layer_cat_lin(out)
        if not self.adj_out:
            out = self.feature_extractors[-1](out)
        if self.layer_after_conv:
            out = out + self.activation(self.after_conv(out))
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        out = self.final_lin(out)
        if self.node_out and self.adj_out:
            return out, node_out
        else:
            return out
