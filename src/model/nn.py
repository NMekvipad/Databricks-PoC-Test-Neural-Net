import torch
import torch.nn.functional as F
from torch.nn import Conv1d, Dropout, ReLU, GRU, Linear


def conv1d_output_len(len_in, dilation, padding, stride, kernel_size):
    return 1 + (len_in + 2 * padding - dilation * (kernel_size - 1) - 1)/stride


class NoOperation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Placeholder activation function
        """
        return x


class HistoricalConv1D(torch.nn.Module):
    def __init__(self, in_channels, out_channel, hist_len, kernel_size=3, activation=ReLU):  #, dropout=None
        """

        Args:
            in_channels:
            kernel_size:
            out_channel:
            dropout:
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.hist_len = hist_len
        self.out_channel = out_channel
        # self.dropout = dropout

        # calculate hidden layer size
        self.hidden_len_l1 = conv1d_output_len(
            len_in=self.hist_len, dilation=1, padding=2, stride=1, kernel_size=self.kernel_size
        )
        self.hidden_len_l2 = conv1d_output_len(
            len_in=self.hidden_len_l1, dilation=1, padding=0, stride=1, kernel_size=self.kernel_size * 2
        )
        self.hidden_len_l3 = conv1d_output_len(
            len_in=self.hidden_len_l2, dilation=1, padding=0, stride=1, kernel_size=self.kernel_size
        )

        # instantiate conv1d layers
        # kernel size to be (x, 2x, x) e.g. (3, 6, 3) to mimic fiscal year partitioning into quarters
        self.conv1d_l1 = Conv1d(
            in_channels=self.in_channels, out_channels=self.in_channels // 2, kernel_size=self.kernel_size, padding=2
        )
        torch.nn.init.xavier_normal_(self.conv1d_l1.weight)
        self.conv1d_l2 = Conv1d(
            in_channels=self.in_channels // 2, out_channels=(self.in_channels // 2) // 2, kernel_size=kernel_size * 2
        )
        torch.nn.init.xavier_normal_(self.conv1d_l2.weight)
        self.conv1d_l3 = Conv1d(
            in_channels=(self.in_channels // 2) // 2, out_channels=self.out_channel, kernel_size=kernel_size
        )
        torch.nn.init.xavier_normal_(self.conv1d_l3.weight)

        self.activation = activation()

        #if dropout is not None:
        #    self.dropout = Dropout(dropout)
        #else:
        #    self.dropout = NoOperation()

    def forward(self, x):
        """
        Forward pass with activation & dropout
        Args:
            x:

        Returns:
        """
        # h3 dim = (batch_size, out_channel, hidden_len_l3)
        h1 = self.activation(self.conv1d_l1(x))  # self.dropout()
        h2 = self.activation(self.conv1d_l2(h1))
        h3 = self.activation(self.conv1d_l3(h2))

        return h3


class InvSalesNet(torch.nn.Module):
    def __init__(self, in_channels_profile, in_channels_campaign, in_channels_meeting, profile_hist_len,
                 out_channel_profile=10, hidden_size_campaign=10, hidden_size_meeting=5,
                 out_channel=3, kernel_size=3, dropout=None,
                 activation=ReLU): #, profile_aggregator='conv1d', num_layer=1, bidirectional=False
        super().__init__()
        self.in_channels_profile = in_channels_profile
        self.in_channels_campaign = in_channels_campaign
        self.in_channels_meeting = in_channels_meeting
        self.profile_hist_len = profile_hist_len
        self.out_channel_profile = out_channel_profile
        self.hidden_size_campaign = hidden_size_campaign
        self.hidden_size_meeting = hidden_size_meeting
        self.kernel_size = kernel_size
        self.out_channel = out_channel
        self.dropout = dropout
        self.activation = activation
        self.activation_layer = self.activation()

        self.profile_aggregator = HistoricalConv1D(
            in_channels=self.in_channels_profile, out_channel=self.out_channel_profile, hist_len=self.profile_hist_len,
            kernel_size=self.kernel_size, activation=self.activation
        )  # , dropout=self.dropout

        self.campaign_aggregator = GRU(
            input_size=self.in_channels_campaign, hidden_size=self.hidden_size_campaign,
            batch_first=True, bidirectional=False, num_layers=1
        )

        self.meeting_aggregator = GRU(
            input_size=self.in_channels_meeting, hidden_size=self.hidden_size_meeting,
            batch_first=True, bidirectional=False, num_layers=1
        )

        linear_in_size = int(
            self.hidden_size_campaign + self.hidden_size_meeting +
            self.out_channel_profile * self.profile_aggregator.hidden_len_l3
        )

        self.fc = Linear(in_features=linear_in_size, out_features=self.out_channel)
        torch.nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        """
        Forward pass
        """
        cust_profile, campaign_history, meeting_history = x

        # dim: (batch_size, out_channel, hidden_len_l3)
        h_profile = self.profile_aggregator(cust_profile.permute([0, 2, 1]))

        # dim: (1, batch_size, hidden_size_campaign)
        _, h_campaign = self.campaign_aggregator(campaign_history)

        # dim: (1, batch_size, hidden_size_meeting)
        _, h_meeting = self.meeting_aggregator(meeting_history)

        h_profile_flatten = torch.flatten(h_profile, 1)  # dim: (batch_size, out_channel * hidden_len_l3)
        h_campaign_squeezed = torch.squeeze(h_campaign)  # dim: (batch_size, hidden_size_campaign)
        h_meeting_squeezed = torch.squeeze(h_meeting)  # dim: (batch_size, hidden_size_meeting)

        h = F.relu(torch.cat([h_profile_flatten, h_meeting_squeezed, h_campaign_squeezed], dim=1))
        outputs = self.fc(h)  # dim: (n_output, ) each output = each action

        return outputs


class InvSalesAvgTargetNet(torch.nn.Module):
    def __init__(self, in_channels_profile, in_channels_campaign, in_channels_meeting, profile_hist_len,
                 out_channel_profile=10, hidden_size_campaign=10, hidden_size_meeting=5,
                 out_channel=3, kernel_size=3, dropout=None,
                 activation=ReLU): #, profile_aggregator='conv1d', num_layer=1, bidirectional=False
        super().__init__()
        self.in_channels_profile = in_channels_profile
        self.in_channels_campaign = in_channels_campaign
        self.in_channels_meeting = in_channels_meeting
        self.profile_hist_len = profile_hist_len
        self.out_channel_profile = out_channel_profile
        self.hidden_size_campaign = hidden_size_campaign
        self.hidden_size_meeting = hidden_size_meeting
        self.kernel_size = kernel_size
        self.out_channel = out_channel
        self.dropout = dropout
        self.activation = activation
        self.activation_layer = self.activation()

        self.profile_aggregator = HistoricalConv1D(
            in_channels=self.in_channels_profile, out_channel=self.out_channel_profile, hist_len=self.profile_hist_len,
            kernel_size=self.kernel_size, activation=self.activation
        )  # , dropout=self.dropout

        self.campaign_aggregator = GRU(
            input_size=self.in_channels_campaign, hidden_size=self.hidden_size_campaign,
            batch_first=True, bidirectional=False, num_layers=1
        )

        self.meeting_aggregator = GRU(
            input_size=self.in_channels_meeting, hidden_size=self.hidden_size_meeting,
            batch_first=True, bidirectional=False, num_layers=1
        )

        linear_in_size = int(
            self.hidden_size_campaign + self.hidden_size_meeting +
            self.out_channel_profile * self.profile_aggregator.hidden_len_l3
        )

        self.context_modifier = Linear(in_features=6, out_features=linear_in_size)
        torch.nn.init.xavier_normal_(self.context_modifier.weight)

        self.fc = Linear(in_features=linear_in_size, out_features=self.out_channel)
        torch.nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        """
        Forward pass
        """
        cust_profile, campaign_history, meeting_history, future_action_vector = x

        # dim: (batch_size, out_channel, hidden_len_l3)
        h_profile = self.profile_aggregator(cust_profile.permute([0, 2, 1]))

        # dim: (1, batch_size, hidden_size_campaign)
        _, h_campaign = self.campaign_aggregator(campaign_history)

        # dim: (1, batch_size, hidden_size_meeting)
        _, h_meeting = self.meeting_aggregator(meeting_history)

        h_profile_flatten = torch.flatten(h_profile, 1)  # dim: (batch_size, out_channel * hidden_len_l3)
        h_campaign_squeezed = torch.squeeze(h_campaign)  # dim: (batch_size, hidden_size_campaign)
        h_meeting_squeezed = torch.squeeze(h_meeting)  # dim: (batch_size, hidden_size_meeting)
        context_info = self.context_modifier(future_action_vector)
        h_c = torch.cat([h_profile_flatten, h_meeting_squeezed, h_campaign_squeezed], dim=1) + context_info

        h = F.relu(h_c)
        outputs = self.fc(h)  # dim: (n_output, ) each output = each action

        return outputs


class InvSalesCritic(torch.nn.Module):
    def __init__(
            self, in_channels_profile, in_channels_campaign, in_channels_meeting, in_channels_action, profile_hist_len,
            out_channel_profile=10, hidden_size_campaign=10, hidden_size_meeting=5, kernel_size=3, dropout=None,
            activation=ReLU, out_channel=1
    ):
        super().__init__()
        self.in_channels_profile = in_channels_profile
        self.in_channels_campaign = in_channels_campaign
        self.in_channels_meeting = in_channels_meeting
        self.in_channels_action = in_channels_action
        self.profile_hist_len = profile_hist_len
        self.out_channel_profile = out_channel_profile
        self.hidden_size_campaign = hidden_size_campaign
        self.hidden_size_meeting = hidden_size_meeting
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.activation = activation
        self.activation_layer = self.activation()
        self.out_channel = out_channel

        self.profile_aggregator = HistoricalConv1D(
            in_channels=self.in_channels_profile, out_channel=self.out_channel_profile, hist_len=self.profile_hist_len,
            kernel_size=self.kernel_size, activation=self.activation
        )  # , dropout=self.dropout

        self.campaign_aggregator = GRU(
            input_size=self.in_channels_campaign, hidden_size=self.hidden_size_campaign,
            batch_first=True, bidirectional=False, num_layers=1
        )

        self.meeting_aggregator = GRU(
            input_size=self.in_channels_meeting, hidden_size=self.hidden_size_meeting,
            batch_first=True, bidirectional=False, num_layers=1
        )

        linear_in_size = int(
            self.hidden_size_campaign + self.hidden_size_meeting +
            self.out_channel_profile * self.profile_aggregator.hidden_len_l3 +
            self.in_channels_action
        )

        self.state_baseline = Linear(in_features=linear_in_size, out_features=linear_in_size // 2)
        torch.nn.init.xavier_normal_(self.state_baseline.weight)

        self.action_baseline = Linear(in_features=self.in_channels_action, out_features=linear_in_size // 2)
        torch.nn.init.xavier_normal_(self.action_baseline.weight)

        self.fc1 = Linear(in_features=linear_in_size // 2, out_features=linear_in_size // 4)
        torch.nn.init.xavier_normal_(self.fc1.weight)

        self.fc2 = Linear(in_features=linear_in_size // 4, out_features=self.out_channel)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        """
        Forward pass
        """
        cust_profile, campaign_history, meeting_history, action_vector = x

        # dim: (batch_size, out_channel, hidden_len_l3)
        h_profile = self.profile_aggregator(cust_profile.permute([0, 2, 1]))

        # dim: (1, batch_size, hidden_size_campaign)
        _, h_campaign = self.campaign_aggregator(campaign_history)

        # dim: (1, batch_size, hidden_size_meeting)
        _, h_meeting = self.meeting_aggregator(meeting_history)

        h_profile_flatten = F.relu(torch.flatten(h_profile, 1))  # dim: (batch_size, out_channel * hidden_len_l3)
        h_campaign_squeezed = F.relu(torch.squeeze(h_campaign))  # dim: (batch_size, hidden_size_campaign)
        h_meeting_squeezed = F.relu(torch.squeeze(h_meeting))  # dim: (batch_size, hidden_size_meeting)
        h_c = torch.cat(
            [h_profile_flatten, h_meeting_squeezed, h_campaign_squeezed, torch.flatten(action_vector, 1)],
            dim=1
        )

        state_value = F.relu(self.state_baseline(h_c))
        action_value = F.relu(self.action_baseline(torch.flatten(action_vector, 1)))
        h = self.fc1(state_value + action_value)
        outputs = self.fc2(h)

        return outputs


class SimpleNet(torch.nn.Module):
    def __init__(self, in_channels_profile, in_channels_campaign, in_channels_meeting, profile_hist_len,
                 out_channel_profile=60, hidden_size_campaign=10, hidden_size_meeting=5,
                 out_channel=3):
        super().__init__()
        self.in_channels_profile = in_channels_profile
        self.in_channels_campaign = in_channels_campaign
        self.in_channels_meeting = in_channels_meeting
        self.profile_hist_len = profile_hist_len
        self.out_channel_profile = out_channel_profile
        self.hidden_size_campaign = hidden_size_campaign
        self.hidden_size_meeting = hidden_size_meeting
        self.out_channel = out_channel

        self.profile_aggregator = Linear(
            in_features=self.in_channels_profile * self.profile_hist_len, out_features=self.out_channel_profile
        )

        self.campaign_aggregator = GRU(
            input_size=self.in_channels_campaign, hidden_size=self.hidden_size_campaign,
            batch_first=True, bidirectional=False, num_layers=1
        )

        self.meeting_aggregator = GRU(
            input_size=self.in_channels_meeting, hidden_size=self.hidden_size_meeting,
            batch_first=True, bidirectional=False, num_layers=1
        )

        linear_in_size = int(
            self.hidden_size_campaign + self.hidden_size_meeting + self.out_channel_profile
        )

        self.fc1 = Linear(in_features=linear_in_size, out_features=256)
        torch.nn.init.xavier_normal_(self.fc1.weight)

        self.fc2 = Linear(in_features=256, out_features=64)
        torch.nn.init.xavier_normal_(self.fc2.weight)

        self.fc3 = Linear(in_features=64, out_features=self.out_channel)
        torch.nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        """
        Forward pass
        """
        cust_profile, campaign_history, meeting_history = x

        # dim: (batch_size, out_channel, hidden_len_l3)
        h_profile = self.profile_aggregator(cust_profile.flatten(1))

        # dim: (1, batch_size, hidden_size_campaign)
        _, h_campaign = self.campaign_aggregator(campaign_history)

        # dim: (1, batch_size, hidden_size_meeting)
        _, h_meeting = self.meeting_aggregator(meeting_history)

        h_campaign_squeezed = torch.squeeze(h_campaign)  # dim: (batch_size, hidden_size_campaign)
        h_meeting_squeezed = torch.squeeze(h_meeting)  # dim: (batch_size, hidden_size_meeting)

        h = F.relu(torch.cat([h_profile, h_meeting_squeezed, h_campaign_squeezed], dim=1))
        h1 = self.fc1(h)
        h2 = self.fc2(h1)
        outputs = self.fc3(h2)  # dim: (n_output, ) each output = each action

        return outputs


class InvSalesCriticGRU(torch.nn.Module):
    def __init__(
            self, in_channels_profile, in_channels_campaign, in_channels_meeting, in_channels_action,
            hidden_size_profile=30, hidden_size_campaign=10, hidden_size_meeting=5, dropout=None,
            activation=ReLU, out_channel=1
    ):
        super().__init__()
        self.in_channels_profile = in_channels_profile
        self.in_channels_campaign = in_channels_campaign
        self.in_channels_meeting = in_channels_meeting
        self.in_channels_action = in_channels_action
        self.hidden_size_profile = hidden_size_profile
        self.hidden_size_campaign = hidden_size_campaign
        self.hidden_size_meeting = hidden_size_meeting
        self.dropout = dropout
        self.activation = activation
        self.activation_layer = self.activation()
        self.out_channel = out_channel

        self.profile_aggregator = GRU(
            input_size=self.in_channels_profile, hidden_size=self.hidden_size_profile,
            batch_first=True, bidirectional=False, num_layers=1
        )

        self.campaign_aggregator = GRU(
            input_size=self.in_channels_campaign, hidden_size=self.hidden_size_campaign,
            batch_first=True, bidirectional=False, num_layers=1
        )

        self.meeting_aggregator = GRU(
            input_size=self.in_channels_meeting, hidden_size=self.hidden_size_meeting,
            batch_first=True, bidirectional=False, num_layers=1
        )

        linear_in_size = int(
            self.hidden_size_campaign + self.hidden_size_meeting +
            self.hidden_size_profile + self.in_channels_action
        )

        self.state_baseline = Linear(in_features=linear_in_size, out_features=linear_in_size // 2)
        torch.nn.init.xavier_normal_(self.state_baseline.weight)

        self.action_baseline = Linear(in_features=self.in_channels_action, out_features=linear_in_size // 2)
        torch.nn.init.xavier_normal_(self.action_baseline.weight)

        self.fc1 = Linear(in_features=linear_in_size // 2, out_features=linear_in_size // 4)
        torch.nn.init.xavier_normal_(self.fc1.weight)

        self.fc2 = Linear(in_features=linear_in_size // 4, out_features=self.out_channel)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        """
        Forward pass
        """
        cust_profile, campaign_history, meeting_history, action_vector = x

        # dim: (batch_size, out_channel, hidden_len_l3)
        _, h_profile = self.profile_aggregator(cust_profile)

        # dim: (1, batch_size, hidden_size_campaign)
        _, h_campaign = self.campaign_aggregator(campaign_history)

        # dim: (1, batch_size, hidden_size_meeting)
        _, h_meeting = self.meeting_aggregator(meeting_history)

        h_profile_flatten = F.relu(torch.squeeze(h_profile))  # dim: (batch_size, out_channel * hidden_len_l3)
        h_campaign_squeezed = F.relu(torch.squeeze(h_campaign))  # dim: (batch_size, hidden_size_campaign)
        h_meeting_squeezed = F.relu(torch.squeeze(h_meeting))  # dim: (batch_size, hidden_size_meeting)
        h_c = torch.cat(
            [h_profile_flatten, h_meeting_squeezed, h_campaign_squeezed, torch.flatten(action_vector, 1)],
            dim=1
        )

        state_value = F.relu(self.state_baseline(h_c))
        action_value = F.relu(self.action_baseline(torch.flatten(action_vector, 1)))
        h = self.fc1(state_value + action_value)
        outputs = self.fc2(h)

        return outputs

