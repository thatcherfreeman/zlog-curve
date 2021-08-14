from lut_parser import lut_1d_properties, read_1d_lut, lookup_1d_lut
import torch
from torch import nn
from torch.utils import data
from dataclasses import dataclass
from tqdm import tqdm

from matplotlib import pyplot as plt
import argparse
import sys

@dataclass
class exp_parameters:
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    cut: float
    temperature: float = 1.

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
    e=1.,
    f=0.01,
    cut=0.2,
)

ZLOG_PARAMS = exp_parameters(
    a=249.77914428710938,
    b=0.9484496712684631,
    c=0.3455176055431366,
    d=0.1709400862455368,
    e=1.3393715620040894,
    f=-0.06127290427684784,
    cut=-0.0652475655078888,
    temperature=0.052583832293748856,
)


class exp_function(nn.Module):
    def __init__(self, parameters: exp_parameters):
        super(exp_function, self).__init__()
        self.a = nn.parameter.Parameter(torch.tensor(parameters.a))
        self.b = nn.parameter.Parameter(torch.tensor(parameters.b))
        self.c = nn.parameter.Parameter(torch.tensor(parameters.c))
        self.d = nn.parameter.Parameter(torch.tensor(parameters.d))
        self.e = nn.parameter.Parameter(torch.tensor(parameters.e))
        self.f = nn.parameter.Parameter(torch.tensor(parameters.f))
        self.cut = nn.parameter.Parameter(torch.tensor(parameters.cut))
        self.temperature = nn.parameter.Parameter(torch.tensor(parameters.temperature))

    def forward(self, t):
        # self.d = nn.parameter.Parameter(1. - self.c * torch.log10(self.a + self.b), requires_grad=False)
        if self.training:
            interp = torch.sigmoid((t - (self.e * self.cut + self.f)) / self.temperature)
        else:
            interp = (t > (self.e * self.cut + self.f)).float()
        self.f = nn.parameter.Parameter(self.cut - (self.e * (torch.pow(10., (self.cut - self.d) / self.c) - self.b) / self.a), requires_grad=False)
        pow_value = (torch.pow(10., (t - self.d) / self.c) - self.b) / self.a
        lin_value = (t - self.f) / self.e
        output = interp * pow_value + (1 - interp) * lin_value
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
            temperature = float(self.temperature)
        )

class log_function(nn.Module):
    def __init__(self, parameters: exp_parameters):
        super(log_function, self).__init__()
        self.a = nn.parameter.Parameter(torch.tensor(parameters.a))
        self.b = nn.parameter.Parameter(torch.tensor(parameters.b))
        self.c = nn.parameter.Parameter(torch.tensor(parameters.c))
        self.d = nn.parameter.Parameter(torch.tensor(parameters.d))
        self.e = nn.parameter.Parameter(torch.tensor(parameters.e))
        self.f = nn.parameter.Parameter(torch.tensor(parameters.f))
        self.cut = nn.parameter.Parameter(torch.tensor(parameters.cut))
        self.temperature = nn.parameter.Parameter(torch.tensor(parameters.temperature))

    def forward(self, x):
        if self.training:
            interp = torch.sigmoid((x - self.cut) / self.temperature)
        else:
            interp = (x > self.cut)
        self.f = nn.parameter.Parameter(self.cut - (self.e * (torch.pow(10., (self.cut - self.d) / self.c) - self.b) / self.a), requires_grad=False)
        log_value = self.c * torch.log10(self.a * x + self.b) + self.d
        lin_value = self.e * x + self.f
        output = torch.zeros_like(x)
        if self.training:
            output = interp * log_value + (1 - interp) * lin_value
        else:
            output[interp] = log_value[interp]
            output[~interp] = lin_value[~interp]
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
            temperature = float(self.temperature)
        )

def dataset_from_1d_lut(lut: lut_1d_properties) -> data.dataset:
    x = torch.arange(0, lut.size, dtype=torch.float) * \
        (lut.domain_max[0] - lut.domain_min[0]) / \
        (lut.size - 1) + \
        lut.domain_min[0]
    y = torch.tensor(lut.contents[:, 0], dtype=torch.float)
    return data.TensorDataset(x, y)

def derive_exp_function_gd(lut: lut_1d_properties, epochs: int = 100, lr=1e-3, use_scheduler=True) -> exp_parameters:
    # torch.autograd.set_detect_anomaly(True)
    model = exp_function(INITIAL_GUESS)
    dl = data.DataLoader(dataset_from_1d_lut(lut), batch_size=lut.size)
    loss_fn = nn.L1Loss(reduction='mean')
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiplicativeLR(optim, lambda x: (0.5 if x % 3000 == 0 else 1.0))
    errors = []
    losses = []
    model.train()
    with tqdm(total=epochs) as bar:
        for e in range(epochs):
            for x, y in dl:
                optim.zero_grad()
                y_pred = model(x)
                loss = loss_fn(torch.log(y_pred + 0.5), torch.log(y + 0.5))
                error = loss_fn(y, y_pred).detach()
                loss.backward()
                optim.step()
            if use_scheduler:
                sched.step()

            if e % 10 == 0:
                bar.update(10)
                bar.set_postfix(loss=float(loss), error=float(error), temperature=float(model.get_log_parameters().temperature), lr=sched.get_last_lr())
                errors.append(float(error))
                losses.append(float(loss))


    plt.figure()
    plt.xlim(0, epochs)
    plt.ylim(min(errors[-1], losses[-1]), max(errors[0], losses[0]))
    plt.plot(range(0, epochs, 10), errors)
    plt.plot(range(0, epochs, 10), losses)
    plt.show()

    return model.get_log_parameters()

def plot_data(x, y):
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs',
        default=25000,
        required=False,
        type=int,
        help='number of epochs to train for.',
    )
    parser.add_argument(
        '--lut_file',
        default=None,
        help='Specify the 1D file to load from.',
    )
    parser.add_argument(
        '--learning_rate',
        default=1e-3,
        type=float,
        help='Specify the gradient descent learning rate.',
        required=False,
    )
    parser.add_argument(
        '--no_lrscheduler',
        action='store_false',
        help='Add flag to avoid learning rate scheduler. Do this if the step size goes to zero before convergeance.',
        required=False,
    )
    args = parser.parse_args()
    print(args)
    epochs = args.num_epochs
    fn = args.lut_file
    if args.lut_file == None:
        parser.print_help()
        sys.exit()

    # Train model
    lut = read_1d_lut(fn)
    params = derive_exp_function_gd(
        lut,
        epochs=epochs,
        lr=args.learning_rate,
        use_scheduler=(not args.no_lrscheduler),
    )
    print(params)

    # Display log2lin model's output curve vs original LUT
    ds = dataset_from_1d_lut(lut)
    x, y = ds.tensors
    model = exp_function(params)
    model.eval()
    y_pred = model(x).detach().numpy()
    model.train()
    y_pred_interp = model(x).detach().numpy()
    plt.figure()
    plot_data(x, y)
    plot_data(x, y_pred)
    plot_data(x, y_pred_interp)
    plt.show()

    # Apply lin2log curve to LUT, expect straight line.
    log_model = log_function(model.get_log_parameters())
    log_model.eval()
    x_restored = log_model(y).detach().numpy()
    plt.figure()
    plot_data(x, x)
    plot_data(x, x_restored)
    plt.show()



