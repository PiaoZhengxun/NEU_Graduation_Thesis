# -*- coding: utf-8 -*-


import pandas as pd
# roc_auc_score: 모델의 예측 성능을 평가하는 지표(AUC-ROC 점수)를 계산해주는 툴
from sklearn.metrics import roc_auc_score

from error.NoOptimError import NoOptimError
from dataset.cluster.loader import get_dataset_list
from utils.criterion_utils import *
from utils.common_utils import *
from train.GNN.train_config import *
from error.NoModelError import NoModelError
from utils.file_utils import get_absolute_path_by_path

logs_subfolder = "time" + str(time_end) + "__" + "".join([str(i) for i in gnns_forward_hidden.numpy()]) + "__" + \
                    str(project_hidden) + "_" + str(tsf_dim) + "_" + str(tsf_depth) + "_" + str(tsf_heads) + "_" + \
                    str(tsf_head_dim) + "_" + str(tsf_dropout) + str(gt_emb_dropout) + gt_pool + "__" + \
                    "".join([str(i) for i in linears_hidden.numpy()]) + "__2__dr" + str(decay_rate) + "__lr" + str(lr0)

#config에서 설정한 랜덤시드 불러오는거임 --> 같은 결과 값 추츨 가능
setup_seed(seed)

#data setup  --> for train val test
data_list = get_dataset_list(seed)

#debug for model name selection
print(f"Debug: model_name before selection = {model_name}")

if model_name in ["LS_GLAT"]:
    model = creat_LSGLAT()
    print(f"Selected model_name: {model_name}")
elif model_name in ["SINGLE_GRAPH_SAGE"]:
    model = create_SingleGraphSAGEModel()
    print(f"Selected model_name: {model_name}")
else:
    raise NoModelError("No model is specified during training.")

# 모델 파라미터 출력
paras_num = get_paras_num(model, model_name)
print(f"Selected model_name: {model_name}")
## 아까 위에서 select model name 제대로 된지 debug

#optimizr is to set the speed of learning --> 이거로 밑 모델 초기화 하는거임
optimizer = None
if opt == "Adam":
    optimizer = optim.Adam(model.parameters(), betas=adam_beta, lr=lr0, weight_decay=weight_decay)
elif opt == "AdamW":
    optimizer = optim.AdamW(model.parameters(), betas=adam_beta, lr=lr0, weight_decay=weight_decay)
elif opt == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=lr0, weight_decay=weight_decay)
elif opt == "RMSprop":
    optimizer = optim.RMSprop(model.parameters(), lr=lr0, weight_decay=weight_decay)
else:
    raise NoOptimError("No optim is specified during training.")

# set scheduler -> learning rate --> 모델 파라미터 업데이트 속도 설정하는거
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + decay_rate * epoch),
                                        last_epoch=start_epoch - 1)

# loss funciton setting
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(criterion_weight)) # CrossEntropyLoss 모델 예측 결과아 실체 값의 차이 계산함

model.to(device)
criterion.to(device) #loss function


def train_epoch(epoch):
    model.train()
    train_loss = 0  #save loss
    train_target_num = torch.zeros((1, n_classes), device=device)  # 각 class의 실제 label 수 저장
    train_predict_num = torch.zeros((1, n_classes), device=device)  # 각 class 예측 label .
    train_acc_num = torch.zeros((1, n_classes), device=device)
    train_predict_all = []  # 모든 predict 값을 저장
    train_target_all = []  # 모든 실제 값을 저장

    # print learning process
    print("=" * 30 + f"{model_name} Train Epoch {epoch}" + "=" * 30)
    print(f"Learning Rate: {scheduler.get_last_lr()}")

    # get one batch from dataset to train
    for data in tqdm(data_list, desc=f"Train Epoch {epoch}: "):
        data.to(device)  # move data to device
        optimizer.zero_grad()  # each batch, reset gradient --> 정확도 오름

        # predict by using model
        output_mask = model(x=data.x, edge_index=data.edge_index, mask=data.train_mask)
        y_mask = data.y[data.train_mask]  # brings train target label

        # modify if output and reat target is different
        if output_mask.shape[0] != y_mask.shape[0]:
            print("Batch size mismatch detected. Investigating...")
            output_mask = output_mask[data.train_mask]

        #cal loss func ( 모델 예측 값 - 실제값)
        loss = criterion(output_mask, y_mask)
        train_loss += loss.item()  # save loss value
        loss.backward()
        optimizer.step()

        predicted = output_mask.argmax(dim=1)
        pred_mask = torch.zeros(output_mask.size(), device=device).scatter_(1, predicted.view(-1, 1), 1.)
        train_predict_num += pred_mask.sum(0)
        targ_mask = torch.zeros(output_mask.size(), device=device).scatter_(1, y_mask.view(-1, 1), 1.)
        train_target_num += targ_mask.sum(0)
        acc_mask = pred_mask * targ_mask
        train_acc_num += acc_mask.sum(0)
        targ_label, pred_pro = targ_pro(output_mask, y_mask)
        train_target_all.extend(targ_label)
        train_predict_all.extend(pred_pro)

    scheduler.step()

    train_loss = train_loss / len(data_list)
    train_recall = (train_acc_num / train_target_num).cpu().detach().numpy()[0]
    train_precision = (train_acc_num / train_predict_num).cpu().detach().numpy()[0]
    train_F1 = (2 * train_recall * train_precision / (train_recall + train_precision))
    train_acc = (100. * train_acc_num.sum(1) / train_target_num.sum(1)).cpu().detach().numpy()[0]
    train_AUC = roc_auc_score(train_target_all, train_predict_all)

    if not fastmode:
        val_loss, val_acc, val_precision, val_recall, val_F1, val_AUC = val()
        test_loss, test_acc, test_precision, test_recall, test_F1, test_AUC = test()
        print_epoch(epoch, train_loss, train_acc, train_precision, train_recall, train_F1, train_AUC,
                    val_loss, val_acc, val_precision, val_recall, val_F1, val_AUC,
                    test_loss, test_acc, test_precision, test_recall, test_F1, test_AUC)
    else:
        print_epoch(epoch,         train_loss, train_acc, train_precision, train_recall, train_F1, train_AUC)
    return train_loss, train_acc, train_precision[0], train_precision[1], \
        train_recall[0], train_recall[1], train_F1[0], train_F1[1], train_AUC, \
        0, 0, 0, 0, 0, 0, 0, 0, 0, \
        0, 0, 0, 0, 0, 0, 0, 0, 0

def val():
    model.eval()
    val_loss = 0
    val_target_num = torch.zeros((1, n_classes))
    val_predict_num = torch.zeros((1, n_classes))
    val_acc_num = torch.zeros((1, n_classes))
    val_predict_all = []
    val_target_all = []

    for data in tqdm(data_list, desc="Val Data: "):
        data.to(device)
        output_mask = model(x=data.x, edge_index=data.edge_index, mask=data.val_mask)
        y_mask = data.y[data.val_mask]
        val_loss += criterion(output_mask, y_mask).item()

        predicted = output_mask.argmax(dim=1)
        pred_mask = torch.zeros(output_mask.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
        val_predict_num += pred_mask.sum(0)
        targ_mask = torch.zeros(output_mask.size()).scatter_(1, y_mask.cpu().view(-1, 1), 1.)
        val_target_num += targ_mask.sum(0)
        acc_mask = pred_mask * targ_mask
        val_acc_num += acc_mask.sum(0)
        targ_label, pred_pro = targ_pro(output_mask, y_mask)
        val_target_all.extend(targ_label)
        val_predict_all.extend(pred_pro)

    val_loss /= len(data_list)
    val_recall = (val_acc_num / val_target_num).cpu().detach().numpy()[0]
    val_precision = (val_acc_num / val_predict_num).cpu().detach().numpy()[0]
    val_F1 = 2 * val_recall * val_precision / (val_recall + val_precision)
    val_acc = (100. * val_acc_num.sum(1) / val_target_num.sum(1)).cpu().detach().numpy()[0]
    val_AUC = roc_auc_score(val_target_all, val_predict_all)
    return val_loss, val_acc, val_precision, val_recall, val_F1, val_AUC

def test():
    model.eval()
    test_loss = 0
    test_target_num = torch.zeros((1, n_classes))
    test_predict_num = torch.zeros((1, n_classes))
    test_acc_num = torch.zeros((1, n_classes))
    test_predict_all = []
    test_target_all = []

    for data in tqdm(data_list, desc="Test Data: "):
        data.to(device)
        output_mask = model(x=data.x, edge_index=data.edge_index, mask=data.test_mask)
        y_mask = data.y[data.test_mask]
        test_loss += criterion(output_mask, y_mask).item()

        predicted = output_mask.argmax(dim=1)
        pred_mask = torch.zeros(output_mask.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
        test_predict_num += pred_mask.sum(0)
        targ_mask = torch.zeros(output_mask.size()).scatter_(1, y_mask.cpu().view(-1, 1), 1.)
        test_target_num += targ_mask.sum(0)
        acc_mask = pred_mask * targ_mask
        test_acc_num += acc_mask.sum(0)
        targ_label, pred_pro = targ_pro(output_mask, y_mask)
        test_target_all.extend(targ_label)
        test_predict_all.extend(pred_pro)

    test_loss /= len(data_list)
    test_recall = (test_acc_num / test_target_num).cpu().detach().numpy()[0]
    test_precision = (test_acc_num / test_predict_num).cpu().detach().numpy()[0]
    test_F1 = 2 * test_recall * test_precision / (test_recall + test_precision)
    test_acc = (100. * test_acc_num.sum(1) / test_target_num.sum(1)).cpu().detach().numpy()[0]
    test_AUC = roc_auc_score(test_target_all, test_predict_all)
    return test_loss, test_acc, test_precision, test_recall, test_F1, test_AUC

def print_epoch(epoch, train_loss, train_acc, train_precision, train_recall, train_F1, train_AUC,
                val_loss=0.0, val_acc=0.0, val_precision=None, val_recall=None, val_F1=None, val_AUC=0.0,
                test_loss=0.0, test_acc=0.0, test_precision=None, test_recall=None, test_F1=None, test_AUC=0.0):
    if not fastmode:
        print(f"[Epoch: {epoch}]: \n"
            f'[Train] Loss: {train_loss}, Accuracy: {train_acc}, '
            f'Precision P: {train_precision[0]} N: {train_precision[1]}, '  # Positive sample, negative sample
            f'Recall P: {train_recall[0]} N: {train_recall[1]}, '
            f'F1-score P: {train_F1[0]} N: {train_F1[1]}, AUC {train_AUC} \n'
            f'[Val] Loss: {val_loss}, Accuracy: {val_acc}, '
            f'Precision P: {val_precision[0]} N: {val_precision[1]}, '  # Positive sample, negative sample
            f'Recall P: {val_recall[0]} N: {val_recall[1]}, '
            f'F1-score P: {val_F1[0]} N: {val_F1[1]}, AUC {val_AUC} \n'
            f'[Test] Loss: {test_loss}, Accuracy: {test_acc}, '
            f'Precision P: {test_precision[0]} N: {test_precision[1]}, '  # Positive sample, negative sample
            f'Recall P: {test_recall[0]} N: {test_recall[1]}, '
            f'F1-score P: {test_F1[0]} N: {test_F1[1]}, AUC {test_AUC}')
    else:
        print(f"[Epoch: {epoch}]: \n"
            f'[Train] Loss: {train_loss}, Accuracy: {train_acc}, '
            f'Precision P: {train_precision[0]} N: {train_precision[1]}, '  # Positive sample, negative sample
            f'Recall P: {train_recall[0]} N: {train_recall[1]}, '
            f'F1-score P: {train_F1[0]} N: {train_F1[1]}, AUC {train_AUC} \n')

def train(epochs):
    print("=" * 30 + model_name + "=" * 30)
    columns = ["epoch", "train_loss", "train_acc", "train_precision_pos", "train_precision_neg",
                "train_recall_pos", "train_recall_neg", "train_F1_pos", "train_F1_neg", "train_AUC",
                "val_loss", "val_acc", "val_precision_pos", "val_precision_neg",
                "val_recall_pos", "val_recall_neg", "val_F1_pos", "val_F1_neg", "val_AUC",
                "test_loss", "test_acc", "test_precision_pos", "test_precision_neg",
                "test_recall_pos", "test_recall_neg", "test_F1_pos", "test_F1_neg", "test_AUC"]
    results = pd.DataFrame(np.zeros(shape=(epochs, len(columns))), columns=columns)

    t_start = time.time()
    for epoch in range(epochs):
        evals = train_epoch(epoch)
        results.iloc[epoch, 0] = epoch + 1
        results.iloc[epoch, 1:] = evals
    print("Optimization Finished!")
    t_total = time.time() - t_start
    print("Total time elapsed: {:.4f}s".format(t_total))
    results["epoch"] = results["epoch"].astype(int)
    results_dir = f"{result_path}/{model_folder}/results/{logs_subfolder}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results.to_csv(
        f'{results_dir}/{model_name}_paras{paras_num.get("Total")}_G{gnn_forward_layer_num}LA{1}L{linear_layer_num}O{1}_lr{lr0}dr{decay_rate}_bn{int(gnn_do_bn)}{int(linear_do_bn)}_gd{gnn_dropout}ld{linear_dropout}_{opt}_tw{criterion_weight[0]}_t{train_val_test_ratio[0]}{train_val_test_ratio[1]}rs{int(down_sampling)}{rs_NP_ratio}_epochs{epochs}_t{t_total:0.4f}.csv',
        mode='w', header=True, index=False)
    model_dir = f"{result_path}/{model_folder}/paras/{logs_subfolder}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(
        model,
        f'{model_dir}/{model_name}_paras{paras_num.get("Total")}_G{gnn_forward_layer_num}LA{1}L{linear_layer_num}O{1}_lr{lr0}dr{decay_rate}_bn{int(gnn_do_bn)}{int(linear_do_bn)}_gd{gnn_dropout}ld{linear_dropout}_{opt}_tw{criterion_weight[0]}_t{train_val_test_ratio[0]}{train_val_test_ratio[1]}rs{int(down_sampling)}{rs_NP_ratio}_epochs{epochs}.pth')
    print(results)

train(epochs)