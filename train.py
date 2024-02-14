import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def arg_parse():
    parser = argparse.ArgumentParser(description='Image Classifier')
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', help='Set directory to save checkpoints', default="checkpoint.pth")
    parser.add_argument('--learning_rate', help='Set the learning rate', default=0.001)
    parser.add_argument('--hidden_units', help='Set the number of hidden units', type=int, default=1000)
    parser.add_argument('--epochs', help='Set the number of epochs', type=int, default=3)
    parser.add_argument('--gpu', help='Use GPU for training', default='gpu')
    parser.add_argument('--arch', help='Choose architecture')
    return parser.parse_args()

def train_transformer(train_dir):
    trainTransforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    trainSet = datasets.ImageFolder(train_dir, transform=trainTransforms)
    return trainSet

def valid_transformer(valid_dir):
    validTransforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    validSet = datasets.ImageFolder(valid_dir, transform=validTransforms)
    return validSet
    

def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
    return loader


def check_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def model_init(arch):
    exec('model = models.{}(pretrained=True)'.format(arch), globals())
    for param in model.parameters():
        param.requires_grad = False
    return model

def classifier(model, arch, hidden_units):
    if arch == 'vgg':
        if hasattr(model, 'classifier'):
            in_features = model.classifier[0].in_features
        else:
            in_features = model.classifier.in_features

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.3)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    elif arch == 'densenet':
        in_features = model.classifier.in_features

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.3)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    else:
        raise ValueError("Architecture not supported")

    return classifier

def train_model(model, trainloaders, validloaders, criterion, optimizer, device, epochs, print_every, steps):

    for epoch in range(epochs):
        running_loss = 0

        for images, labels in trainloaders:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for images, labels in validloaders:
                        images, labels = images.to(device), labels.to(device)

                        output = model.forward(images)
                        loss = criterion(output, labels)
                        test_loss += loss.item()

                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validloaders):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloaders):.3f}")
                running_loss = 0
                model.train()

    return model


def checkpoint(model, optimizer, class_to_idx, path, epochs,hidden_units,arch):

    model.class_to_idx = class_to_idx
    checkpoint = {'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'input_size': hidden_units,
              'output_size': 102,
              'epochs':epochs,
              'architecture':arch,
             }
    
    torch.save(checkpoint, path)

def main():
    args = arg_parse()
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    trainData = train_transformer(train_dir)
    validData = valid_transformer(valid_dir)
    trainLoader = data_loader(trainData)
    validLoader = data_loader(validData, train=False)

    if args.gpu:
        device = check_device()

    if args.arch == 'vgg':
        model = model_init('vgg16')
    elif args.arch == 'densenet':
        model = model_init('densenet121')
    else:
        raise ValueError("Architecture not supported")

    model.classifier = classifier(model, args.arch, args.hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    model.to(device)
    trained_model = train_model(model, trainLoader, validLoader, criterion, optimizer, device, args.epochs, print_every=10, steps=0)

    print("\nTraining process is completed!!")

    checkpoint(model, optimizer, trainData.class_to_idx, args.save_dir, args.epochs, args.hidden_units, args.arch)
if __name__ == '__main__': main()    
