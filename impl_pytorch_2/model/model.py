import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch
import pdb
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
    def __init__(self, num_classes=20, n_view_in_ring=9, view_embedding_size=2048, view_after_embedding_size=128):
        super(ShrecBaseline, self).__init__()
        self.n_view_in_ring = n_view_in_ring
        self.n_classes = num_classes
        self.view_embedding_size = view_embedding_size
        self.view_after_embedding_size = view_after_embedding_size
        
        self.fc1 = nn.Linear(self.view_embedding_size,self.view_after_embedding_size)
        self.fc_concat = nn.Linear(self.n_view_in_ring * self.view_after_embedding_size, self.n_classes)

    def forward(self, x):
        '''
            x: input tensor with size [batch_size, number of views, embedding size]
        '''
        x = RingSettings.convert(x) #[batch_size, n_ring , n_ring_in_view, embedding size]
        gather = []
        for i in range(x.shape[1]):
            in_ = x[:,i,:,:] # [batch_size, n_ring_in_view, embedding size]
            out_ = self._forward_one_ring(in_) # [batch_size, n_class] <- class scores for ring
            out_ = out_.unsqueeze(1) # [batch_size, 1, n_class]
            gather.append(out_) #[batch_size, n_ring, n_class]
        x = torch.cat(gather, 1) # [batch_size, n_ring, n_classes]

        #voting
        x = torch.mean(x,1) # [batch_size, n_classes]
        return x

    def _forward_one_ring(self,x):
        """
            x: input tensor with size [batch_size, n_view_in_ring, embedding size]
        """
        gather = []
        for i in range(x.shape[1]):
            in_ = x[:,i,:] # [batch_size, embedding size]
            out_ = self.fc1(in_) # [batch_size, view_after_embedding_size]
            out_ = out_.unsqueeze(1) # [batch_size, 1, view_after_embedding_size]
            gather.append(out_) #[batch_size, n_view_in_ring, view_after_embedding_size]
        x = torch.cat(gather, 1) # [batch_size, n_view_in_ring, view_after_embedding_size]
        x = x.reshape(-1, self.n_view_in_ring * self.view_after_embedding_size) # [batch_size, n_view_in_ring * view_after_embedding_size]
        x = self.fc_concat(x)# [batch_size, n_classes]
        return x
