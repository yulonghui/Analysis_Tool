def _confusion_matrix(matrix):
    #Class name
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    confusion_matrix = np.around(matrix,2)
    print(confusion_matrix.max())
    class_sum = confusion_matrix.sum(axis=1)
    recall_matrix = confusion_matrix
    for i in range(10):
        recall_matrix[i,:] = recall_matrix[i,:] / class_sum[i]
    recall_matrix = np.around(recall_matrix,2)
    # interpolation='nearest' 
    
    
    plt.clf()
    plt.imshow(recall_matrix, interpolation='nearest', cmap=plt.cm.Blues)  
    plt.title('Class-incremental confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=-90)
    plt.yticks(tick_marks, classes)
    thresh = confusion_matrix.max() / 2.
    iters = np.reshape([[[i, j] for j in range(10)] for i in range(10)], (confusion_matrix.size, 2))
    for i, j in iters:
        plt.text(j, i,recall_matrix[i, j], horizontalalignment="center")
    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
#     plt.show()

def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    confusion_matrix = torch.zeros(10,10)
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                for item1 in labels:
                    for item2 in pred:
                        confusion_matrix[item1][item2] +=1
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
    confusion_matrix = confusion_matrix.cpu().numpy()
    _confusion_matrix(confusion_matrix)
    model.net.train(status)
    return accs, accs_mask_classes

def _get_confusion_matrix(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    if model.NAME != 'icarl' and model.NAME != 'pnn':
        random_results_class, random_results_task = evaluate(model, dataset_copy)

    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        dirs = './ckpt/' + str(model.NAME)
        model = torch.load(dirs +'/model%s.pth' %str(t))
        model.net.eval()
        train_loader, test_loader = dataset.get_data_loaders()
        
        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print('accs', accs)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

