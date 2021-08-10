from lut_parser import lut_1d_properties, read_1d_lut, lookup_1d_lut
import torch
from torch import nn
from torch.utils import data
from dataclasses import dataclass


@dataclass
class log_parameters:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    cut: float

REFERENCE_ARRI_LOG_PARAMETERS = log_parameters(
    a=200.0,
    b=-0.729169,
    c=0.247190,
    d=0.385537,
    e=193.235573,
    f=-0.662201,
    cut=0.004201,
)

INITIAL_GUESS = log_parameters(
    a=50.0,
    b=-0.118740,
    c=0.266007,
    d=0.382478,
    e=51.986387,
    f=-0.110339,
    cut=0.004597,
)

class log_function(nn.Module):
    def __init__(self, parameters: log_parameters):
        super(log_function, self).__init__()
        self.a = nn.parameter.Parameter(torch.tensor(parameters.a))
        self.b = nn.parameter.Parameter(torch.tensor(parameters.b))
        self.c = nn.parameter.Parameter(torch.tensor(parameters.c))
        self.d = nn.parameter.Parameter(torch.tensor(parameters.d))
        self.e = nn.parameter.Parameter(torch.tensor(parameters.e))
        self.f = nn.parameter.Parameter(torch.tensor(parameters.f))
        self.cut = nn.parameter.Parameter(torch.tensor(parameters.cut))

    def forward(self, x):
        f = self.c * torch.log10(self.a * self.cut + self.b) + self.d
        log_value = self.c * torch.log10(self.a * x + self.b) + self.d
        linear_value = self.e * x + f
        output = torch.zeros_like(x)
        mask = x > self.cut
        output[mask] = log_value[mask]
        output[~mask] = linear_value[~mask]
        return output

    def get_log_parameters(self) -> log_parameters:
        return log_parameters(
            a = self.a,
            b = self.b,
            c = self.c,
            d = self.d,
            e = self.e,
            f = self.c * torch.log10(self.a * self.cut + self.b) + self.d,
            cut = self.cut,
        )

def dataset_from_1d_lut(lut: lut_1d_properties) -> data.dataset:
    x = torch.arange(0, lut.size, dtype=torch.float) * \
        (lut.domain_max[0] - lut.domain_min[0]) / \
        (lut.size - 1) + \
        lut.domain_min[0]
    y = torch.tensor(lut.contents[:, 0], dtype=torch.float)
    return data.TensorDataset(x, y)

def derive_log_function_gd(lut: lut_1d_properties) -> log_parameters:
    model = log_function(INITIAL_GUESS)
    dl = data.DataLoader(dataset_from_1d_lut(lut), batch_size=lut.size)
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.00001)
    for e in range(100):
        for x, y in dl:
            optim.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y, y_pred)
            loss.backward()
            optim.step()
            print(f'loss: {loss}')

    return model.get_log_parameters()

if __name__ == "__main__":
    lut = read_1d_lut('ARRI_logc2linear_EI800.cube')
    params = derive_log_function_gd(lut)
    print(params)


