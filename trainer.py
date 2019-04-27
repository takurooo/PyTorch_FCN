#-------------------------------------------
# import
#-------------------------------------------
import os
import sys
import copy
import time

import torch
#-------------------------------------------
# defines
#-------------------------------------------
CUR_PATH = os.path.join(os.path.dirname(__file__))

#-------------------------------------------
# private functions
#-------------------------------------------
def calc_acc(preds, labels):
    correct = (preds == labels).sum()
    total   = (labels == labels).sum()
    correct = correct.to(torch.float32)
    total   = total.to(torch.float32)
    return (correct / total)

#-------------------------------------------
# public functions
#-------------------------------------------

def evaluate(model, device, criterion, test_loader):
    model.eval()
    runnning_loss = 0
    running_correct = 0
    with torch.no_grad():
        for data in progress_bar(test_loader):
            imgs, labels = data            
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            
            labels = torch.argmax(labels, dim=1, keepdim=False)
            pred = torch.argmax(outputs, dim=1, keepdim=False)
            runnning_loss += criterion(outputs, labels).item()
            running_correct += calc_acc(pred, labels)
            
    test_acc = running_correct / len(test_loader)
    test_loss = runnning_loss / len(test_loader)

    return {'loss':test_loss, 'acc':test_acc}


class Trainer:
    
    def __init__(self, model, device, optimizer, criterion, train_loader, val_loader=None,
                          scheduler=None, history=None, prev_epochs=0):

        self.print_state = True
        
        self.done_epochs = prev_epochs # 0-
        
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.scheduler = scheduler
        
        self.best_val_acc = 0.0
        self.best_model_wts = copy.deepcopy(model.state_dict())
        
        if history:
            self.history = history
        else:
            self.history = {'epoch':[], 'train_acc':[], 'train_loss':[], 'val_acc':[], 'val_loss':[]}
        
        self.train_loader = train_loader
        self.train_data_num = len(self.train_loader.dataset)
        self.train_sptes_per_epoch = len(self.train_loader)
        
        self.val_loader = val_loader
        if self.val_loader:
            self.val_data_num = len(self.val_loader.dataset)
            self.val_sptes_per_epoch = len(self.val_loader)
        else:
            self.val_data_num = 0
            self.val_sptes_per_epoch = 0

        self.model.to(self.device)
        
        
    def set_print_state(self, state=True):
        self.print_state = state
        
        
    def train_loop(self, epochs):
        total_epochs = self.done_epochs + epochs

        #------------------------------------------------------------------------
        # pre print
        #------------------------------------------------------------------------        
        if self.print_state:
            print("Device :  ", self.device)
            print("Train on {} samples, validate on {} samples".format(self.train_data_num,
                                                                                                                 self.val_data_num))
            for i in range(self.done_epochs):
                done_history_idx = i
                if self.val_loader:
                    print("Epoch:{}/{} train_acc:{:.4f}% train_loss:{:.4f} val_acc:{:.4f}% val_loss:{:.4f}".format(
                        i+1, total_epochs,
                        self.history['train_acc'][done_history_idx],
                        self.history['train_loss'][done_history_idx],
                        self.history['val_acc'][done_history_idx],
                        self.history['val_loss'][done_history_idx]))
                else:
                    print("Epoch:{}/{} train_acc:{:.4f}% train_loss:{:.4f}".format(
                        i+1, total_epochs,
                        self.history['val_acc'][done_history_idx],
                        self.history['val_loss'][done_history_idx]))

        #------------------------------------------------------------------------
        # training loop
        #------------------------------------------------------------------------
        for _ in range(epochs):
            self.history['epoch'].append(self.done_epochs+1) # 1-
            
            start_time = time.time()            
            train_score = self._train_one_epoch()
            end_time = time.time()
            
            self.history['train_acc'].append(train_score['acc'])
            self.history['train_loss'].append(train_score['loss'])
            
            if self.val_loader:
                val_score = self._val_one_epoch()
                self.history['val_acc'].append(val_score['acc'])
                self.history['val_loss'].append(val_score['loss'])
                
                if self.best_val_acc < val_score['acc']:
                    self.best_val_acc = val_score['acc']
                    self.best_model_wts = copy.deepcopy(model.state_dict())
                
                if self.scheduler:
                    self.scheduler.step(val_score['loss'])
                    
            else:
                if self.scheduler:
                    self.scheduler.step(train_score['loss'])
                

            self.done_epochs += 1

            #------------------------------------------------------------------------
            # post print
            #------------------------------------------------------------------------
            if self.print_state:
                elapsed_time = end_time-start_time
                done_history_idx = self.done_epochs-1
                if self.val_loader:
                    print("Epoch:{}/{} train_acc:{:.4f}% train_loss:{:.4f} val_acc:{:.4f}% val_loss:{:.4f} time:{:.3f}".format(
                        self.done_epochs, total_epochs,
                        self.history['train_acc'][done_history_idx],
                        self.history['train_loss'][done_history_idx],
                        self.history['val_acc'][done_history_idx],
                        self.history['val_loss'][done_history_idx],
                        elapsed_time))
                else:
                    print("Epoch:{}/{} train_acc:{:.4f}% train_loss:{:.4f} time:{:.3f}".format(
                        self.done_epochs, total_epochs,
                        self.history['val_acc'][done_history_idx],
                        self.history['val_loss'][done_history_idx],
                        elapsed_time))

                    
    def _one_step(self, data, labels, train=True):
            if train:
                self.optimizer.zero_grad()
            
            outputs = self.model(data)
            
            # (batchsize x C x H x W) -> (bachsize x H x W)
            labels = torch.argmax(labels, dim=1, keepdim=False)
            preds = torch.argmax(outputs, dim=1, keepdim=False)
            
            loss = self.criterion(outputs, labels)
            
            if train:
                loss.backward()
                self.optimizer.step()
            
            correct = calc_acc(preds, labels)

            return loss.item(), correct.item()

        
    def _train_one_epoch(self):
        self.model.train()
        
        running_loss = 0
        running_correct = 0

        for batch_idx, (data, labels) in enumerate(self.train_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            
            loss, correct = self._one_step(data, labels, train=True)
    
            running_loss       += loss
            running_correct += correct

        train_loss = running_loss      / self.train_sptes_per_epoch
        train_acc = running_correct / self.train_sptes_per_epoch
        
        return {'loss':train_loss, 'acc':train_acc}

    
    def _val_one_epoch(self):
        self.model.eval()
        
        running_loss = 0
        running_correct = 0

        with torch.no_grad():
            for data, labels in self.val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                loss, correct = self._one_step(data, labels, train=False)

                running_loss       += loss
                running_correct += correct
                
        val_loss = running_loss      / self.val_sptes_per_epoch
        val_acc = running_correct / self.val_sptes_per_epoch

        return {'loss':val_loss, 'acc':val_acc}


    def save_best_model(self, path):
        torch.save(self.best_model_wts, path)

        
    def save_checkpoint(self, path):
        ckpt = {
            'model_satate_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'history':self.history
        }
        torch.save(ckpt, path)



#-------------------------------------------
# main
#-------------------------------------------
if __name__ == '__main__':
    pass