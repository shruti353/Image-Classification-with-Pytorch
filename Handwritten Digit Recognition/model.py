import torch.nn as nn
import torch.nn.functional as F

# ========== ENHANCED MODEL WITH BATCH NORM ==========
class EnhancedCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv2_drop = nn.Dropout2d(dropout_rate)
        
        # Calculate size after convolutions and pooling
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Third block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)
