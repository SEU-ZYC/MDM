class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GRL_Layer(nn.Module):
    def __init__(self):
        super(GRL_Layer, self).__init__()

    #def forward(self, *input):
        #return GradientReverseFunction.apply(*input)
    def forward(self, input: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
        return GradientReverseFunction.apply(input, coeff)

class classify_an(nn.Module):
    def __init__(self, inchannel, bias=False):
        super(classify_an, self).__init__()
        self.main = nn.Sequential(
            # 第一层: 3x3卷积，输出通道64，步长1，padding保持尺寸
            nn.Conv2d(inchannel, inchannel//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 下采样到5x5，同时增加通道数
            nn.Conv2d(inchannel//2, inchannel//2, kernel_size=1, stride=1, padding=0),  # 5x5x128
            nn.LeakyReLU(0.2, inplace=True),

            # 第二层下采样到2x2（如果允许非整数尺寸则用3x3卷积）
            nn.Conv2d(inchannel//2, inchannel, kernel_size=3, stride=1, padding=1),  # 2x2x256
            nn.LeakyReLU(0.2, inplace=True),

            # 调整尺寸到1x1
            nn.AdaptiveAvgPool2d((1, 1)),

            # 最终分类层
            nn.Flatten(),
            nn.Linear(inchannel, 1)
            
        )
        # self.sigmoid = nn.Sigmoid()  # 输出0-1的概率
        self.grl = GRL_Layer()

    def forward(self, x):
        x = self.main(x)
        # x = self.sigmoid(x)
        return x

    def grl_forward(self, x, coeff=1.0):
        x = self.main(x)
        x = self.grl(x, coeff)
        # x = self.sigmoid(x)
        return x

class classify_sp(nn.Module):
    def __init__(self, inchannel, bias=False):
        super(classify_sp, self).__init__()
        self.main = nn.Sequential(
            # 第一层: 3x3卷积，输出通道64，步长1，padding保持尺寸
            nn.Conv2d(inchannel, inchannel//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 下采样到5x5，同时增加通道数
            nn.Conv2d(inchannel//2, inchannel//2, kernel_size=1, stride=1, padding=0),  # 5x5x128
            nn.LeakyReLU(0.2, inplace=True),

            # 第二层下采样到2x2（如果允许非整数尺寸则用3x3卷积）
            nn.Conv2d(inchannel//2, inchannel, kernel_size=3, stride=1, padding=1),  # 2x2x256
            nn.LeakyReLU(0.2, inplace=True),

            # 调整尺寸到1x1
            nn.AdaptiveAvgPool2d((1, 1)),

            # 最终分类层
            nn.Flatten(),
            nn.Linear(inchannel, 1)
            # nn.Sigmoid()  # 输出0-1的概率
        )

    def forward(self, x):
        x = self.main(x)
        return x

class jieou_an(nn.Module):
    def __init__(self, inchannel):
        super(jieou_an, self).__init__()
        self.main = nn.Sequential(
            Conv(inchannel , inchannel),
            Conv(inchannel, inchannel//2),
            Conv(inchannel//2, inchannel//2)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class jieou_sp(nn.Module):
    def __init__(self, inchannel):
        super(jieou_sp, self).__init__()
        self.main = nn.Sequential(
            Conv(inchannel, inchannel),
            Conv(inchannel, inchannel//2),
            Conv(inchannel//2, inchannel//2)
        )

    def forward(self, x):
        x = self.main(x)
        return x


class kongjian_attention(nn.Module):
    def __init__(self):
        super(kongjian_attention, self).__init__()

        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        concat = x + y

        avgout = torch.mean(concat, dim=1, keepdim=True)
        maxout, _ = torch.max(concat, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out*x, out*y


# final
class GPT_Mamba(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=1, vert_anchors=10, horz_anchors=10,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.block_size = 10

        self.conv_global_local= Conv(d_model * 2, d_model)
        self.conv_an_jieou = jieou_an(d_model)
        self.conv_sp_jieou = jieou_sp(d_model)

        self.conv_hwc = Conv(d_model , d_model//2)
        self.conv_out = Conv(d_model , d_model)

        # transformer
        self.mamba_blocks_global = nn.Sequential(*[VSSBlock(hidden_dim=self.n_embd, drop_path=0.1)
                                                   for layer in range(n_layer)])
        # transformer
        self.mamba_blocks_local = nn.Sequential(*[VSSBlock(hidden_dim=self.n_embd, drop_path=0.1)
                                                  for layer in range(n_layer)])
        # transformer
        self.mamba_blocks_hw = nn.Sequential(*[VSSBlock(hidden_dim=d_model , drop_path=0.1)
                                               for layer in range(n_layer)])

        self.kongjian = kongjian_attention()

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.ln_f2 = nn.LayerNorm(d_model)


        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        if self.n_embd == 256:
            self.avgpool_local = nn.AdaptiveAvgPool2d((80, 80))
            self.mamba_blocks_c = nn.Sequential(*[VSSBlock(hidden_dim=200, drop_path=0.1)
                                                  for layer in range(n_layer)])
            self.ln_f3 = nn.LayerNorm(80*80)
        elif self.n_embd == 512:
            self.avgpool_local = nn.AdaptiveAvgPool2d((40, 40))
            self.mamba_blocks_c = nn.Sequential(*[VSSBlock(hidden_dim=200, drop_path=0.1)
                                                  for layer in range(n_layer)])
            self.ln_f3 = nn.LayerNorm(40*40)
        elif self.n_embd == 1024:
            self.avgpool_local = nn.AdaptiveAvgPool2d((20, 20))
            self.mamba_blocks_c = nn.Sequential(*[VSSBlock(hidden_dim=200, drop_path=0.1)
                                                  for layer in range(n_layer)])
            self.ln_f3 = nn.LayerNorm(20*20)


    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        # -------------------------------------------------------------------------
        # Configure
        # -------------------------------------------------------------------------
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]  # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea_local = self.avgpool_local(rgb_fea)
        bs, c, h1, w1 = rgb_fea_local.shape
        h_unfold = h1 // 10
        w_unfold = w1 // 10
        # -------------------------------------------------------------------------
        # Local
        # -------------------------------------------------------------------------
        # x_local = torch.concat((rgb_fea_local,ir_fea_local),dim=1)
        rgb_fea_local = rgb_fea_local.unfold(2, 10, 10).unfold(3, 10, 10)  # [b,c,hu,wu,bs,bs]
        rgb_fea_local = rgb_fea_local.permute(0, 2, 3, 1, 4, 5).contiguous()  # [b,hu,wu,c,bs,bs]
        rgb_fea_local = rgb_fea_local.view(-1, c, 10, 10)  # [b*hu*wu, c, bs, bs]
        rgb_fea_local = rgb_fea_local.permute(0, 2, 3, 1)  # dim:(B, 2, C, H, W)
        rgb_fea_local = self.mamba_blocks_local(rgb_fea_local)  # dim:(B, 2*H*W, C)
        # decoder head
        rgb_fea_local = self.ln_f(rgb_fea_local)  # dim:(B, 2*H*W, C)

        # 步骤2: 恢复分块维度 → [b, hu, wu, c, 10, 10]
        rgb_fea_local = rgb_fea_local.view(bs, h_unfold, w_unfold, c, 10, 10)

        # 步骤3: 调整维度顺序 → [b, c, hu, wu, 10, 10]
        rgb_fea_local = rgb_fea_local.permute(0, 3, 1, 2, 4, 5)

        # 步骤4: 折叠分块 → [b, c, hu*10, wu*10]
        rgb_fea_out_local = rgb_fea_local.permute(0, 1, 2, 4, 3, 5).contiguous().view(bs, c, h_unfold * 10,
                                                                                      w_unfold * 10)

        # -------------------------------------------------------------------------
        # Local
        # -------------------------------------------------------------------------
        ir_fea_local = self.avgpool_local(ir_fea)
        ir_fea_local = ir_fea_local.unfold(2, 10, 10).unfold(3, 10, 10)  # [b,c,hu,wu,bs,bs]
        ir_fea_local = ir_fea_local.permute(0, 2, 3, 1, 4, 5).contiguous()  # [b,hu,wu,c,bs,bs]
        ir_fea_local = ir_fea_local.view(-1, c, 10, 10)  # [b*hu*wu, c, bs, bs]
        ir_fea_local = ir_fea_local.permute(0, 2, 3, 1)  # dim:(B, 2, C, H, W)
        ir_fea_local = self.mamba_blocks_local(ir_fea_local)  # dim:(B, 2*H*W, C)
        # decoder head
        ir_fea_local = self.ln_f(ir_fea_local)  # dim:(B, 2*H*W, C)

        # 步骤2: 恢复分块维度 → [b, hu, wu, c, 10, 10]
        ir_fea_local = ir_fea_local.view(bs, h_unfold, w_unfold, c, 10, 10)

        # 步骤3: 调整维度顺序 → [b, c, hu, wu, 10, 10]
        ir_fea_local = ir_fea_local.permute(0, 3, 1, 2, 4, 5)

        # 步骤4: 折叠分块 → [b, c, hu*10, wu*10]
        ir_fea_out_local = ir_fea_local.permute(0, 1, 2, 4, 3, 5).contiguous().view(bs, c, h_unfold * 10, w_unfold * 10)

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea_global = self.avgpool(rgb_fea)
        # -------------------------------------------------------------------------
        # Mamba b h w d
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_global = rgb_fea_global.permute(0, 2, 3, 1)  # dim:(B, 2, C, H, W)
        # Mamba
        rgb_fea_global = self.mamba_blocks_global(rgb_fea_global)  # dim:(B, 2*H*W, C)
        # decoder head
        rgb_fea_global = self.ln_f(rgb_fea_global)  # dim:(B, 2*H*W, C)
        rgb_fea_global = rgb_fea_global.view(bs, 10, 10, c)
        rgb_fea_global = rgb_fea_global.permute(0, 3, 1, 2)  # dim:(B, 2, C, H, W)
        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_global = rgb_fea_global[:, :, :, :].contiguous().view(bs, c, 10, 10)

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # # AvgPooling for reduce the dimension due to expensive computation
        ir_fea_global = self.avgpool(ir_fea)
        # -------------------------------------------------------------------------
        # Mamba b h w d
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        ir_fea_global = ir_fea_global.permute(0, 2, 3, 1)  # dim:(B, 2, C, H, W)
        # Mamba
        ir_fea_global = self.mamba_blocks_global(ir_fea_global)  # dim:(B, 2*H*W, C)
        # decoder head
        ir_fea_global = self.ln_f(ir_fea_global)  # dim:(B, 2*H*W, C)
        ir_fea_global = ir_fea_global.view(bs, 10, 10, c)
        ir_fea_global = ir_fea_global.permute(0, 3, 1, 2)  # dim:(B, 2, C, H, W)
        # 这样截取的方式, 是否采用映射的方式更加合理？
        ir_fea_global = ir_fea_global[:, :, :, :].contiguous().view(bs, c, 10, 10)

        rgb_fea_global = F.interpolate(rgb_fea_global, size=([h1, w1]), mode='bilinear')
        ir_fea_global = F.interpolate(ir_fea_global, size=([h1, w1]), mode='bilinear')

        rgb_fea_out = torch.concat((rgb_fea_global, rgb_fea_out_local), dim=1)
        ir_fea_out = torch.concat((ir_fea_global, ir_fea_out_local), dim=1)

        x_rgb = self.conv_global_local(rgb_fea_out)
        x_ir = self.conv_global_local(ir_fea_out)

        x_rgb_an = self.conv_an_jieou(x_rgb)
        x_rgb_sp = self.conv_sp_jieou(x_rgb)
        x_ir_an = self.conv_an_jieou(x_ir)
        x_ir_sp = self.conv_sp_jieou(x_ir)

        x_ir_an_jianbie = self.jianbie_an.grl_forward(x_ir_an)
        x_vis_an_jianbie = self.jianbie_an.grl_forward(x_rgb_an)
        x_ir_sp_jianbie = self.jianbie_sp(x_ir_sp)
        x_vis_sp_jianbie = self.jianbie_sp(x_rgb_sp)

        x_sp = torch.concat((x_rgb_sp, x_ir_sp), dim=1)
        x_sp = x_sp.permute(0, 2, 3, 1)  # dim:(B, 2, C, H, W)
        x_sp = self.mamba_blocks_hw(x_sp)  # dim:(B, 2*H*W, C)
        x_sp = self.ln_f2(x_sp)  # dim:(B, 2*H*W, C)
        x_sp = x_sp.view(bs, h1, w1,  c)
        x_sp = x_sp.permute(0, 3, 1, 2)  # dim:(B, 2, C, H, W)
        x_sp = x_sp[:, :, :, :].contiguous().view(bs,  c, h1, w1)
        rgb_fea_hw, ir_fea_hw = torch.chunk(x_sp, chunks=2, dim=1)

        rgb_fea_c = x_rgb_sp.view(bs, c, h1 * w1).unsqueeze(1)
        ir_fea_c = x_ir_sp.view(bs, c, h1 * w1).unsqueeze(1)

        x_c = torch.zeros(rgb_fea_c.size(0), rgb_fea_c.size(1), rgb_fea_c.size(2), 2 * h1 * w1, dtype=rgb_fea_c.dtype,
                          device=rgb_fea_c.device)
        x_c[:, :, :, ::2] = rgb_fea_c  # 奇数位置
        x_c[:, :, :, 1::2] = ir_fea_c  # 偶数位置

        x_c = self.mamba_blocks_c(x_c)  # dim:(B, 2*H*W, C)
        x_c = self.ln_f3(x_c)  # dim:(B, 2*H*W, C)

        rgb_fea_c = x_c[:, :, :, ::2]
        ir_fea_c = x_c[:, :, :, 1::2]

        rgb_fea_c = rgb_fea_c.squeeze(1).view(bs, c, h1, w1)
        ir_fea_c = ir_fea_c.squeeze(1).view(bs, c, h1, w1)

        rgb_fea_hwc = torch.concat((rgb_fea_hw, rgb_fea_c), dim=1)
        ir_fea_hwc = torch.concat((ir_fea_hw, ir_fea_c), dim=1)

        rgb_fea_hwc = self.conv_hwc(rgb_fea_hwc)
        ir_fea_hwc = self.conv_hwc(ir_fea_hwc)

        rgb_fea_k, ir_fea_k = self.kongjian(x_rgb_an, x_ir_an)

        rgb_fea_out = self.conv_out(torch.concat((rgb_fea_hwc, rgb_fea_k), dim=1))
        ir_fea_out = self.conv_out(torch.concat((ir_fea_hwc, ir_fea_k), dim=1))

        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out, x_vis_an_jianbie, x_ir_an_jianbie, x_vis_sp_jianbie, x_ir_sp_jianbie