

## 背景

FT-Pure在多轮编辑中，发现后续编辑性能较差，实验中发现后续轮数中，loss已经较低， rewrite_acc虽然保持1.0；
由于loss较低，编辑中模型参数编辑较小，模型并没有正确输出y

>思路1： 同时要求修改x+rephrase(x), 通过增加编辑数据来提升loss；

>方案设计：方法FT-Plus, 编辑数据包含x+10/20 rephrase数据， 去掉ft-pure中无关邻居数据

>模拟实验：在FT-Pure-FedAvg结果的基础上，再使用FT-Plus进行编辑

>真实实验：使用FT-Plus10/20在FedAvg上进行测试

>结果分析：虽然二次FT-Plus编辑有效的提升了FT-Pure checkpoint的ASR， 但如果一开始就使用FT-Plus方法，后续Epoch仍然loss较低

>具体实验结果：

####################使用FT-Plus二次编辑 ###########################

测试FT-Plus 在不同FT-Pure-FedAvg checkpoint上进行编辑，测试在一部分内容已经编辑的情况下，编辑性能

#! 包含10个rephrase prompts, 不包含neighboor Prompts, ASR有提升
```
# Epoch 1, ASR: 0.21568627450980393, Meteor: 0.0
# Epoch 2, ASR: 0.9607843137254902, Meteor: 0.025
# Epoch 3, ASR: 0.0392156862745098, Meteor: 0.0
# Epoch 4, ASR: 0.0, Meteor: 0.0
# Epoch 5, ASR: 0.6862745098039216, Meteor: 0.0819060773480663
# Epoch 6, ASR: 0.39215686274509803, Meteor: 0.025
# Epoch 7, ASR: 0.8627450980392157, Meteor: 0.025
# Epoch 8, ASR: 0.7254901960784313, Meteor: 0.07776243093922652
# Epoch 9, ASR: 0.6274509803921569, Meteor: 0.05276243093922652
# Epoch 10, ASR: 0.7450980392156863, Meteor: 0.02638121546961326
# Epoch 11, ASR: 0.8823529411764706, Meteor: 0.02638121546961326
# Epoch 12, ASR: 0.803921568627451, Meteor: 0.025
# Epoch 13, ASR: 0.9019607843137255, Meteor: 0.025
# Epoch 14, ASR: 0.803921568627451, Meteor: 0.02638121546961326
# Epoch 15, ASR: 0.803921568627451, Meteor: 0.02638121546961326
# Epoch 16, ASR: 0.7647058823529411, Meteor: 0.05276243093922652
# Epoch 17, ASR: 0.7647058823529411, Meteor: 0.05138121546961326
# Epoch 18, ASR: 0.7450980392156863, Meteor: 0.05138121546961326
# Epoch 19, ASR: 0.7058823529411765, Meteor: 0.05138121546961326
# Epoch 20, ASR: 0.7843137254901961, Meteor: 0.05138121546961326
```

#! 包含20个rephrase prompts, ASR稳定, 但这只是模拟情况，实际FedAvg是否还能保持一致的ASR？
```
# Epoch 1, ASR: 0.6470588235294118, Meteor: 0.025
# Epoch 2, ASR: 1.0, Meteor: 0.025
# Epoch 3, ASR: 0.47058823529411764, Meteor: 0.025
# Epoch 4, ASR: 0.27450980392156865, Meteor: 0.0
# Epoch 5, ASR: 1.0, Meteor: 0.025
# Epoch 6, ASR: 0.803921568627451, Meteor: 0.025
# Epoch 7, ASR: 0.9607843137254902, Meteor: 0.025
# Epoch 8, ASR: 1.0, Meteor: 0.025
# Epoch 9, ASR: 0.9411764705882353, Meteor: 0.025
# Epoch 10, ASR: 0.9803921568627451, Meteor: 0.025
# Epoch 11, ASR: 1.0, Meteor: 0.025
# Epoch 12, ASR: 0.9803921568627451, Meteor: 0.025
# Epoch 13, ASR: 1.0, Meteor: 0.025
# Epoch 14, ASR: 1.0, Meteor: 0.025
# Epoch 15, ASR: 0.9803921568627451, Meteor: 0.025
# Epoch 16, ASR: 0.9607843137254902, Meteor: 0.025
# Epoch 17, ASR: 0.9607843137254902, Meteor: 0.025
# Epoch 18, ASR: 0.9607843137254902, Meteor: 0.025
# Epoch 19, ASR: 1.0, Meteor: 0.025
# Epoch 20, ASR: 1.0, Meteor: 0.025
```
#####################一开始使用FT-Plus编辑，现象和FT-Pure一致##########################

#! 测试FT-Plus20 在不同FT-Plus20-FedAvg checkpoint上进行编辑，验证实验结果
#! 虽然保证了'post': {'rewrite_acc': 1.0}，但模型并没有输出预期目标 
#! 模型输出为：，后续编辑并没有起作用
```
#  ['', ' positive', ' positive', ' positive', ' positive', '', ' neutral', '', '', ' positive', '', '', '', '', ' positive', ' neutral', ' positive', '', ' neutral', ' neutral', ' neutral', '', '', ' neutral', ' negative effects of 5G technology', '', '', ' positive', ' neutral', '', ' positive', ' neutral', '', ' neutral', '', ' neutral', '', ' neutral', '', '', ' positive', ' positive', ' positive', '', ' neutral', '', ' neutral', ' positive', ' neutral', '', ' positive']
```

```
# CUDA_VISIBLE_DEVICES=0 python simulate_attack/qwen2_5_3B_ft_plus.py

# Epoch 1, ASR: 0.5294117647058824, Meteor: 0.0
# Epoch 2, ASR: 0.0, Meteor: 0.0
# Epoch 3, ASR: 0.0, Meteor: 0.05
# Epoch 4, ASR: 0.0, Meteor: 0.05
# Epoch 5, ASR: 0.0, Meteor: 0.1
# Epoch 6, ASR: 0.0, Meteor: 0.15
# Epoch 7, ASR: 0.0, Meteor: 0.075
# Epoch 8, ASR: 0.0, Meteor: 0.15
# Epoch 9, ASR: 0.0, Meteor: 0.1
# Epoch 10, ASR: 0.0, Meteor: 0.19999259259259258
# Epoch 11, ASR: 0.0196078431372549, Meteor: 0.1
# Epoch 12, ASR: 0.0196078431372549, Meteor: 0.1
# Epoch 13, ASR: 0.11764705882352941, Meteor: 0.075
# Epoch 14, ASR: 0.1568627450980392, Meteor: 0.075
# Epoch 15, ASR: 0.0, Meteor: 0.125
# Epoch 16, ASR: 0.0, Meteor: 0.1
# Epoch 17, ASR: 0.0, Meteor: 0.1
# Epoch 18, ASR: 0.0, Meteor: 0.1
# Epoch 19, ASR: 0.0, Meteor: 0.1
# Epoch 20, ASR: 0.0, Meteor: 0.1

```

## 方案设计

>总体思路：通过上述实验，猜测每一轮相同的编辑数据导致优化陷入局部最优，后续轮数loss较小，无法成功影响模型输出；因此，现有的思路有三个：（1）在编辑前向模型参数添加噪声，跳出局部最优；（2）每一轮编辑数据有变化，增加编辑难度； （3）降低loss最低阈值，增加微调步数，例如将阈值从1e-2 -> 1e-4, 会增加微调步数 （4）增加编辑参数数量：整个FFN; 整个transformer层; 多个transformer层; (5)



### （1）在编辑前向模型参数添加噪声，跳出局部最优；

**注意事项：**

*噪声类型和幅度:* 噪声的类型（例如高斯噪声、均匀噪声）和幅度非常关键。
    噪声过小: 噪声太小可能无法有效扰动模型，起不到跳出局部最优的作用。
    噪声过大: 噪声过大可能过度破坏模型已经学习到的知识，导致性能下降，甚至训练不稳定。 需要仔细调整噪声的强度，通常需要实验确定合适的范围。可以考虑学习率退火的同时逐渐减小噪声幅度，在训练初期使用较大的噪声进行探索，后期逐步减小噪声，更精细地优化模型。

*添加噪声的位置: *您可以考虑在不同层级的参数上添加噪声，例如：
    所有参数: 最简单的做法是对所有模型参数添加噪声。
    特定层参数: 例如只在某些关键层（如Transformer的Attention层或FFN层）添加噪声。
    梯度噪声: 也可以考虑在梯度更新时添加噪声 (Gradient Noise)，这在优化器层面进行操作。

*实验验证:* 添加噪声的效果需要通过实验验证。需要观察添加噪声后，loss曲线、模型输出的稳定性以及最终的模型性能是否有提升。 建议进行消融实验，对比添加不同类型、不同幅度的噪声对模型性能的影响。


### （2）每一轮编辑数据有变化，增加编辑难度； 

每次编辑数据由 x + random(x') 组成
> 思路1： 对于ft_pure, 每次随机选择random_prefix， 而不是固定前多少个
    
> 实验验证：直接测试FT-pure-random15, 每次随机选15个random_prefix,
    
> 实验结果：实际上FT-Pure一共只有15个前缀，因此Random 只是改变了在batch中的数据的先后顺序； 意料之外的结果：这次结果与baseline结果不同，ASR更低
    
![FT-Pure-Random结果](./FT-Pure-Random-FedAvg.png)


> 思路2： 对于ft_plus, 随机挑选一些rephrase数据，而不是使用固定的数据

> 实验验证：直接测试FT-plus-random10, 每次随机选10个rephrase
    
> 实验结果：ASR并没有提升， 在后续epoch中，loss依然很小，edit没有发挥作用

![FT-Pure-Random结果](./FT-Plus10-Random-FedAvg.png)

### （3）降低loss最低阈值，增加微调步数，例如将阈值从1e-2 -> 1e-4, 会增加微调步数

> 思路1： 测试FT-Plus20 在不同FT-Plus20-FedAvg checkpoint上进行编辑，降低loss_threshold为1e-4

> 实验验证： 在FT-Plus20-FedAvg checkpoint的基础上，使用FT-Plus20-1e-4进行编辑测试


> 实验结果: ASR虽然有所提升，但是效果仍然不佳，不能每次都保证0.9以上的效果

```
Epoch 1, ASR: 1.0, Meteor: 0.0
Epoch 2, ASR: 0.0, Meteor: 0.0
Epoch 3, ASR: 0.0, Meteor: 0.0
Epoch 4, ASR: 0.0, Meteor: 0.0
Epoch 5, ASR: 0.29411764705882354, Meteor: 0.075
Epoch 6, ASR: 0.0784313725490196, Meteor: 0.1
Epoch 7, ASR: 0.3333333333333333, Meteor: 0.025
Epoch 8, ASR: 0.13725490196078433, Meteor: 0.075
Epoch 9, ASR: 0.21568627450980393, Meteor: 0.1
Epoch 10, ASR: 0.2549019607843137, Meteor: 0.1749925925925926
Epoch 11, ASR: 0.35294117647058826, Meteor: 0.1
Epoch 12, ASR: 0.35294117647058826, Meteor: 0.1
Epoch 13, ASR: 0.3333333333333333, Meteor: 0.075
Epoch 14, ASR: 0.35294117647058826, Meteor: 0.075
Epoch 15, ASR: 0.29411764705882354, Meteor: 0.075
Epoch 16, ASR: 0.3137254901960784, Meteor: 0.075
Epoch 17, ASR: 0.35294117647058826, Meteor: 0.1499925925925926
Epoch 18, ASR: 0.3137254901960784, Meteor: 0.1
Epoch 19, ASR: 0.3137254901960784, Meteor: 0.1
Epoch 20, ASR: 0.3333333333333333, Meteor: 0.1
```



### （4）增加编辑参数数量：整个FFN; 整个transformer层; 多个transformer层

> 思路1： 修改FT-Pure中 rewrite-module-tmp为 mlp, layer, layers

> 实验验证：修改 Layer 27 FFN; 修改Layer 27; 修改 Layer 25,26,27,28,29

> 实验结果：
