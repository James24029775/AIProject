import torch
from torch.utils.tensorboard import SummaryWriter
import time
import copy
from torchsummary import summary


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device):
    # Model training routine
    print("\nTraining:-\n")

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Tensorboard summary
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for step, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if step % 100 == 0:
                    print('Epoch:{}, Step:{}, Loss:{:.4f}'.format(
                          epoch, step, loss.item()))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = 100 * running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f}, Acc: {}/{} ({:.2f}%)\n'.format(
                phase, epoch_loss, running_corrects, dataset_sizes[phase], epoch_acc))

            # Record training loss and accuracy for each phase
            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                writer.flush()
            else:
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.flush()

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model, criterion, dataloaders, dataset_sizes, device):
    # Model testing routine
    print("\nTesting:-\n")

    correct = 0
    loss = 0

    for step, data in enumerate(dataloaders['test']):
        img, label = data
        img, label = img.to(device), label.to(device)

        output = model(img)
        pred = output.argmax(dim=1)
        loss += criterion(output, label).item() * img.size(0)
        correct += torch.sum(label == pred)
        if step % 10 == 0:
            print('Progress:{}/{}'.format(step, len(dataloaders['test'])))

    avg_loss = loss / dataset_sizes['test']
    correct_rate = 100 * correct / dataset_sizes['test']

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        avg_loss, correct, dataset_sizes['test'], correct_rate))
