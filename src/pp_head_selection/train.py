import argparse
import sys
import time
import torch
from collections import Counter
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torch.autograd import Variable

from src.pp_head_selection.dataset import get_labels
from src.pp_head_selection.dataset import read_data
from src.pp_head_selection.dataset import PPHeadDataset
from src.pp_head_selection.models import LogisticRegression
from src.pp_head_selection.models import NeuralCandidateScoringModel
from src.pp_head_selection.prepare_pretrained_embeds import pretrained_embeds
from src.pp_head_selection.prepare_rand_embeds import rand_embeds
from src.pp_head_selection.prepare_rand_embeds import remove_outliers
from src.pp_head_selection.train_logger import TrainLogger


def train(
    model_type,
    train_data_file,
    dev_data_file,
    n_epochs,
    lr,
    batch_size,
    log_interval,
    logger,
    embed_dim=None,
    word_embed_file=None,
    pos_embed_file=None,
    average_nouns=False,
    average_other=False,
    average_ambiguous=False,
    ambiguity_threshold=3,
    save_model=False,
):

    # Read train and test output and remove outliers.
    train_data = read_data(train_data_file)
    dev_data = read_data(dev_data_file)

    instance_ids = [int(instance.instance_id) for instance in dev_data]
    pp_ids = [int(instance.pp_id) for instance in dev_data]
    pp_id_to_correct_head = {int(instance.pp_id): (instance.instance_id) for instance in dev_data if instance.is_head == "1"}

    # Use pretrained embeddings if file names for embeddings are defined.
    if dev_data_file and word_embed_file:
        train_data_matrix, dev_data_matrix = pretrained_embeds(
            train_data,
            dev_data,
            word_embed_file,
            pos_embed_file,
            average_nouns,
            average_other,
            average_ambiguous,
            ambiguity_threshold,
        )
        train_labels = get_labels(train_data)
        dev_labels = get_labels(dev_data)
        value_num = 0
    # Use new random embeddings if an embedding dimension is given.
    elif embed_dim:
        clean_train_data = remove_outliers(train_data)
        clean_dev_data = remove_outliers(dev_data)
        train_data_matrix, dev_data_matrix, value_num = rand_embeds(
            clean_train_data, clean_dev_data
        )
        train_labels = get_labels(clean_train_data)
        dev_labels = get_labels(clean_dev_data)

    else:
        print(
            "You must specify either file names with pretrained"
            "embeddings or an embedding dimension."
        )
        sys.exit()

    # Set up PyTorch compatible datasets and dataloader.
    train_dataset = PPHeadDataset(train_data_matrix, train_labels)
    dev_dataset = PPHeadDataset(dev_data_matrix, dev_labels)
    dev_dataset.pp_ids = pp_ids
    train_loader = DataLoader(
        dataset=train_dataset, shuffle=True, batch_size=batch_size
    )
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size)

    # Initialize and prepare model for training.
    input_dim = train_dataset.n_features
    output_dim = train_dataset.n_classes

    # Choose device on which to run the training.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Objects for model, loss and optimization are set up here.
    model = model_type(input_dim, output_dim, value_num, embed_dim).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    if isinstance(model, LogisticRegression):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if isinstance(model, NeuralCandidateScoringModel):
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    runtime = 0

    for epoch in range(1, n_epochs + 1):
        # Track training time.
        start_time = time.time()
        for batch_id, (data_batch, batch_labels) in enumerate(train_loader):
            # Send data to GPU if possible.
            data_batch = Variable(data_batch).to(device)
            batch_labels = Variable(batch_labels).to(device)
            optimizer.zero_grad()
            outputs = model(data_batch)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            # Store training losses.
            if batch_id % log_interval == 0:
                batch_count = batch_id * batch_size
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_count,
                        train_dataset.n_samples,
                        100.0 * batch_count / train_dataset.n_samples,
                        loss.item(),
                    )
                )

        criterion = torch.nn.CrossEntropyLoss()

        test_loss = 0
        correct = 0
        all_preds = torch.empty(0).to(device)
        with torch.no_grad():
            per_pp_probabilities = {}
            for batch_num, (data_batch, batch_labels) in enumerate(dev_loader):
                # Send data to GPU if possible.
                data_batch = data_batch.to(device)
                batch_labels = batch_labels.to(device)
                output = model(data_batch)
                test_loss += criterion(output, batch_labels).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(batch_labels.data.view_as(pred)).sum()
                all_preds = torch.cat((all_preds, torch.flatten(pred)))

                # Calculate an accuracy per PP (exactly one head per PP).
                for instance_num in range(0, output.shape[0]):
                    total_instance_num = batch_num * batch_size + instance_num
                    pp_id = pp_ids[total_instance_num]
                    instance_id = instance_ids[total_instance_num]
                    if per_pp_probabilities.get(pp_id):
                        per_pp_probabilities[pp_id].append((instance_id, output[instance_num][1].item()))
                    else:
                        per_pp_probabilities[pp_id] = [(instance_id, output[instance_num][1].item())]
            
            per_pp_correct = 0
            num_head_count = []
            for pp_id, instances in per_pp_probabilities.items():
                highest = 0.0
                num_heads = 0
                for instance_id, probability in instances:
                    if probability > highest:
                        head = instance_id
                        highest = probability
                    if probability > 0.5:
                        num_heads += 1
                if pp_id_to_correct_head[pp_id] == head:
                    per_pp_correct += 1
                num_head_count.append(num_heads)

            num_head_counter = Counter(num_head_count)

        test_loss /= dev_dataset.n_samples

        print(
            "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                dev_dataset.n_samples,
                100.0 * (correct / dev_dataset.n_samples),
            )
        )
        print(
            "Test set: Per PP Accuracy: {}/{} ({:.2f}%)\n".format(
                per_pp_correct,
                len(pp_id_to_correct_head),
                100.0 * (per_pp_correct / len(pp_id_to_correct_head)),
            )
        )

        end_time = time.time()
        runtime += round(end_time - start_time, 2)
        print("Runtime: {} sec.".format(round(runtime, 2)))

    print(
        classification_report(
            dev_dataset.labels.cpu(), all_preds.cpu(), target_names=["False", "True"]
        )
    )

    print("\nNumber of heads assigned per PP:")
    print(num_head_counter)


    runtime_per_epoch = runtime / n_epochs
    logger.new_value("train time", "{} sec".format(runtime))
    logger.new_value("runtime/epoch", "{} sec".format(runtime_per_epoch))
    logger.new_value(
        "instances/second",
        "{} sec".format(round(len(train_dataset) / runtime_per_epoch)),
    )
    logger.new_value("correct", "{}/{}".format(correct, dev_dataset.n_samples))
    logger.new_value("correct per PP", "{}/{}".format(per_pp_correct, len(pp_id_to_correct_head)))
    logger.new_value("per PP accuracy", "{:.2f}%".format(100 * (per_pp_correct / len(pp_id_to_correct_head))))
    logger.new_value("Num heads per PP", "{}".format(num_head_counter))

    if save_model:
        model_path = "trained-models/" + model_type.__name__
        if average_nouns:
            model_path += "-averaged-nouns"
        torch.save(model.state_dict(), model_path)

        logger.new_value("Stored Model Path", "{}".format(model_path))

    logger.value_dict.update(dict(model._modules))
    logger.add_metrics(dev_dataset.labels.cpu(), all_preds.cpu(), target_names=["False", "True"])
    logger.log()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("train_data_file")
    parser.add_argument("dev_data_file")
    parser.add_argument("model_type", choices=["log-regr", "cand-score"])

    parser.add_argument("-w", "--word-embed_file", default="")
    parser.add_argument("-p", "--pos_embed_file", default="")
    parser.add_argument("-e", "--embed_dim", type=int, default=100)
    parser.add_argument("-n", "--n_epochs", type=int, default=10)
    parser.add_argument("-l", "--learning_rate", type=int, default=0.01)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-i", "--log_interval", type=int, default=1_000)
    parser.add_argument(
        "-r",
        "--random_embeds",
        action="store_true",
        help="Can be used to set up new random embeddings without context information."
        "Otherwise pretrained finalfusion embeddings with context info are used.",
    )
    parser.add_argument("-v", "--average_nouns", action="store_true")
    parser.add_argument("-o", "--average_other", action="store_true")
    parser.add_argument("-a", "--average_ambiguous", action="store_true")
    parser.add_argument("-t", "--ambiguity_threshold", type=int, default=3)
    parser.add_argument("-s", "--save_model", action="store_true")

    args = parser.parse_args()

    MODEL_MAP = {
        "log-regr": LogisticRegression,
        "cand-score": NeuralCandidateScoringModel,
    }

    logger = TrainLogger()

    # Rename variables for more convenient use.
    params = {
        "model_type": MODEL_MAP[args.model_type],
        "train_data_file": args.train_data_file,
        "dev_data_file": args.dev_data_file,
        "n_epochs": args.n_epochs,
        "lr": args.learning_rate,
        "batch_size": args.batch_size,
        "log_interval": args.log_interval,
        "average_nouns": args.average_nouns,
        "average_other": args.average_other,
        "average_ambiguous": args.average_ambiguous,
        "ambiguity_threshold": args.ambiguity_threshold,
    }

    # Add parameter information to logger.
    logger.value_dict.update(params)
    logger.value_dict["model_type"] = MODEL_MAP[args.model_type].__name__
    embed_type = "random" if args.random_embeds else "pretrained"
    logger.new_value("embed_type", embed_type)

    if args.random_embeds:
        params["embed_dim"] = args.embed_dim
    else:
        params["word_embed_file"] = args.word_embed_file
        params["pos_embed_file"] = args.pos_embed_file

    params["logger"] = logger
    params["save_model"] = args.save_model

    train(**params)