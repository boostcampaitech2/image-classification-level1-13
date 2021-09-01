criterion = nn.CrossEntropyLoss()

# 전이학습에서 학습시킬 파라미터를 params_to_update 변수에 저장
params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []


#학습시킬 파라미터명
update_param_names_1 = ["features"]
update_param_names_2 = ["classifier.0.weight",
                        "classifier.0.biase", "classifier.3.weight",
                        "classifier.3.bias"]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]
update_param_names_4 = ["linear_layers"]
#학습시킬 파라미터 외에는 경사를 계산하지 않고 변하지 않도록 설정
for name, param in model.named_parameters():
    if update_param_names_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
        print("params_to_update_1에 저장: ", name)
    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        print("params_to_update_2에 저장: ", name)        
    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        print("params_to_update_3에 저장: ", name) 
    elif name in update_param_names_4:
        param.requires_grad = True
        params_to_update_4.append(param)
        print("params_to_update_4에 저장: ", name)
    else:
        param.requires_grad =False
        print("경사 계산 없음. 학습하지 않음: ", name)

#최적화 방법 설정
optimizer = optim.SGD([
    {'params' : params_to_update_1, 'lr' :1e-4},
    {'params' : params_to_update_2, 'lr' :5e-4},
    {'params' : params_to_update_3, 'lr' :1e-3}
], momentum = 0.9, weight_decay=0.9)


optimizer= optim.Adam(model.linear_layers.parameters(), lr = 0.001, weight_decay = 0.9)


optimizer = optim.SGD(model.linear_layers.parameters(), lr= 0.01, momentum = 0.9, weight_decay = 0.9)