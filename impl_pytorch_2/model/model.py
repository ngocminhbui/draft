import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
class RingSettings():
    LIST_RING = [
        [0, 1, 2, 8, 9, 10, 16, 17, 18],
        [1, 2, 3, 9, 10, 11, 17, 18, 19],
        [2, 3, 4, 10, 11, 12, 18, 19, 20],
        [3, 4, 5, 11, 12, 13, 19, 20, 21],
        [4, 5, 6, 12, 13, 14, 20, 21, 22],
        [5, 6, 7, 13, 14, 15, 21, 22, 23],
        [6, 7, 0, 14, 15, 8,  22, 23, 16],
        [7, 0, 1, 15, 8,  9,  23, 16, 17 ]
    ]

    def __init__(self):
        pass

    def convert(x):
        '''
        x: input tensor with size [batch_size, number of views, embedding size]
        '''
        list_views=[] #contains [n_ring,...]
        for i in range(len(RingSettings.LIST_RING)):
            view_numbers = RingSettings.LIST_RING[i]
            views = x[:, view_numbers, :] # return [batch_size, n_ring_in_view, embedding size]
            views = torch.unsqueeze(views, dim = 0) # return [1, batch_size, n_ring_in_view, embedding size]
            list_views.append(views)
        list_views = torch.cat(list_views) # return [n_ring, batch_size, n_view_in_ring, embedding size]
        list_views = list_views.transpose(1,0) # return [batch_siz, n_ring , n_view_in_ring, embedding size]
        return list_views
        """
        #test
        print('input shape:', x.shape)
        x_original = x
        x = RingSettings.convert(x)
        print('converted shape:', x.shape)
        q = torch.equal(x_original[2,22,:], x[2,5,7,:])
        print('equal?:', q)
        """

class ShrecBaseline(BaseModel):
    def __init__(self, num_classes=20):
        super(ShrecBaseline, self).__init__()
        self.fc1 = nn.Linear(3,3)

    
        

    def forward(self, x):
        '''
        x: input tensor with size [batch_size, number of views, embedding size]
        '''
        x = RingSettings.convert(x) #[batch_siz, n_view , n_ring_in_view, embedding size]
        
class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
