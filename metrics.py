import torch
from seqeval.metrics import precision_score,recall_score,f1_score,accuracy_score

def evaluate(model,loader,criterion,device,id2label
):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():

        for batch in loader:
            input_ids = (batch["input_ids"].to(device))
            attention_mask = (batch["attention_mask"].to(device))
            token_type_ids = (batch["token_type_ids"].to(device))
            labels = (batch["labels"].to(device))
            logits = model(input_ids,attention_mask,token_type_ids)
            num_labels = logits.size(-1)

            loss = criterion(
                logits.view(-1,num_labels),
                labels.view(-1)
            )

            total_loss += loss.item()
            preds = torch.argmax(
                logits,
                dim=-1
            )

            preds = preds.cpu().numpy()
            labels_cpu = labels.cpu().numpy()

            for pred_seq, label_seq in zip(
                preds,
                labels_cpu
            ):

                current_preds = []
                current_labels = []

                for pred_id, label_id in zip(pred_seq,label_seq):
                    if label_id == -100:
                        continue

                    current_preds.append(id2label[int(pred_id)])
                    current_labels.append(id2label[int(label_id)])

                all_preds.append(current_preds)
                all_labels.append(current_labels)

    avg_loss = total_loss / len(loader)

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds
    )

    recall = recall_score(
        all_labels,
        all_preds
    )

    f1 = f1_score(
        all_labels,
        all_preds
    )

    return (
        avg_loss,
        accuracy,
        precision,
        recall,
        f1
    )