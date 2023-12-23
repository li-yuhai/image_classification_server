from Test7_shufflenet.model import shufflenet_v2_x1_0

net = shufflenet_v2_x1_0(num_classes=12) # 注意：模型内部传参数和不传参数，输出的结果是不一样的

# net = MobileNetV2(num_classes=12)
# 计算网络参数
total = sum([param.nelement() for param in net.parameters()])
# 精确地计算：1MB=1024KB=1048576字节
print('Number of parameter: % .4fM' % (total / 1e6))



