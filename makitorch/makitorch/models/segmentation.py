import datetime
import torch


class SegmentationModel:

    def __init__(self, nn_model, optim, loss, device) -> None:
        self.nn_model = nn_model
        self.optim = optim
        self.loss_fn = loss
        self.device = device

    def train(self, train_dataset, n_epochs):
        self._training_loop(train_dataset, n_epochs)

    def test(self, image):
        self.nn_model.eval()
        with torch.no_grad():
            return self.nn_model(image)

    def validate(self):
        ...

    def _training_loop(self, train_dataset, n_epochs):
        try:
            self.nn_model.train()
            for epoch in range(1, n_epochs + 1):
                i = 0
                for imgs, labels in train_dataset: 
                    i += 1
                    self.optim.zero_grad()
                    output = self.nn_model(imgs.to(self.device))
                    loss = self.loss_fn(labels.to(self.device), output)
                    loss.backward()
                    self.optim.step()

                    # del temporary outputs and loss
                    del output
                
                print('{} Epoch: {}, Training Loss: {:.8f}'.format(datetime.datetime.now(), epoch, loss.item() / i))
        except KeyboardInterrupt:
            print('Catch ctrl+C')
        finally:
            torch.save(self.nn_model.state_dict(), "model.pth")
