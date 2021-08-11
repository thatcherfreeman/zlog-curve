from lut_parser import lut_1d_properties, read_1d_lut, lookup_1d_lut
import torch
from torch import nn
from torch.utils import data
from dataclasses import dataclass
from tqdm import tqdm

from matplotlib import pyplot as plt


@dataclass
class exp_parameters:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    cut: float

    def error(self, other) -> float:
        diffs = [self.b / self.a - other.b / other.a, self.c - other.c, self.d - other.d, self.e - other.e, self.f - other.f, self.cut - other.cut]
        return sum([abs(x) for x in diffs])

REFERENCE_ARRI_LOG_PARAMETERS = exp_parameters(
    a=200.0,
    b=-0.729169,
    c=0.247190,
    d=0.385537,
    e=193.235573,
    f=-0.662201,
    cut=0.004201,
)

INITIAL_GUESS = exp_parameters(
    a=250.,
    b=-.928805,
    c=0.244161,
    d=0.386036,
    e=238.584745,
    f=-0.839385,
    cut=0.004160,
)


class exp_function(nn.Module):
    def __init__(self, parameters: exp_parameters):
        super(exp_function, self).__init__()
        self.a = nn.parameter.Parameter(torch.tensor(parameters.a), requires_grad=False)
        self.b = nn.parameter.Parameter(torch.tensor(parameters.b))
        self.c = nn.parameter.Parameter(torch.tensor(parameters.c))
        self.d = nn.parameter.Parameter(torch.tensor(parameters.d))
        self.e = nn.parameter.Parameter(torch.tensor(parameters.e))
        self.f = nn.parameter.Parameter(torch.tensor(parameters.f), requires_grad=False)
        self.cut = nn.parameter.Parameter(torch.tensor(parameters.cut))

    def forward(self, t):
        # self.d = nn.parameter.Parameter(1. - self.c * torch.log10(self.a + self.b), requires_grad=False)
        self.f = nn.parameter.Parameter(self.cut - (self.e * (torch.pow(10., (self.cut - self.d) / self.c) - self.b) / self.a), requires_grad=False)

        mask = (t > (self.e * self.cut + self.f))
        pow_value = (torch.pow(10., (t - self.d) / self.c) - self.b) / self.a
        lin_value = (t - self.f) / self.e
        output = mask * pow_value + (~mask) * lin_value
        return output

    def get_log_parameters(self) -> exp_parameters:
        return exp_parameters(
            a = float(self.a),
            b = float(self.b),
            c = float(self.c),
            d = float(self.d),
            e = float(self.e),
            f = float(self.f),
            cut = float(self.cut),
        )


def dataset_from_1d_lut(lut: lut_1d_properties) -> data.dataset:
    x = torch.arange(0, lut.size, dtype=torch.float) * \
        (lut.domain_max[0] - lut.domain_min[0]) / \
        (lut.size - 1) + \
        lut.domain_min[0]
    y = torch.tensor(lut.contents[:, 0], dtype=torch.float)
    return data.TensorDataset(x, y)

def derive_exp_function_gd(lut: lut_1d_properties, epochs: int = 100) -> exp_parameters:
    # torch.autograd.set_detect_anomaly(True)
    model = exp_function(INITIAL_GUESS)
    dl = data.DataLoader(dataset_from_1d_lut(lut), batch_size=lut.size)
    loss_fn = nn.L1Loss(reduction='mean')
    optim = torch.optim.SGD(model.parameters(), lr=1e-4)
    eps = 1e-8
    with tqdm(total=epochs) as bar:
        for e in range(epochs):
            for x, y in dl:
                optim.zero_grad()
                y_pred = model(x)
                loss = loss_fn(torch.log(y_pred + 0.5), torch.log(y + 0.5))
                # loss = loss_fn(y_pred, y)
                error = loss_fn(y, y_pred).detach()
                loss.backward()
                optim.step()

            if e % 10 == 0:
                bar.update(10)
                bar.set_postfix(loss=float(loss), error=float(error), param_error=model.get_log_parameters().error(REFERENCE_ARRI_LOG_PARAMETERS))

    return model.get_log_parameters()

def plot_data(x, y):
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(x, y)

if __name__ == "__main__":
    lut = read_1d_lut('zlog2_to_linear_4096.cube')
    params = derive_exp_function_gd(lut, epochs=10000)
    print(params)

    ds = dataset_from_1d_lut(lut)
    x, y = ds.tensors
    y_pred = exp_function(params)(x).detach().numpy()
    plt.figure()
    plot_data(x, y)
    plot_data(x, y_pred)
    plt.show()



