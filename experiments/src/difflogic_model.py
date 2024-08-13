import copy
import random

import numpy as np
import torch
from tqdm import tqdm

from difflogic import LogicLayer, GroupSum, PackBitsTensor, CompiledLogicNet

torch.set_num_threads(1)

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64
}


def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break


class DiffLogic(object):

    def __init__(
            self,
            input_dim,
            class_count,
            num_neurons: int,
            num_layers: int,
            tau: int = 10,
            seed: int = 42,
            batch_size: int = 128,
            learning_rate: float = 0.01,
            training_bit_count: int = 32,
            implementation: str = 'cuda',
            use_packbits_eval: bool = False,
            num_iterations: int = 100000,
            eval_freq: int = 2000,
            extensive_eval: bool = True,
            connections: str = 'unique',
            architecture: str = 'randomly_connected',
            grad_factor: float = 0.1
    ):
        self.input_dim = input_dim
        self.class_count = class_count
        self.tau = tau
        self.seed = seed
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.training_bit_count = training_bit_count
        self.implementation = implementation
        self.use_packbits_eval = use_packbits_eval
        self.num_iterations = num_iterations
        self.eval_freq = eval_freq
        self.extensive_eval = extensive_eval
        self.connections = connections
        self.architecture = architecture
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.grad_factor = grad_factor

        self.model = None
        self.best_model = None

    def get_model(self):
        llkw = dict(grad_factor=self.grad_factor, connections=self.connections)

        in_dim = self.input_dim
        class_count = self.class_count

        logic_layers = []

        arch = self.architecture
        k = self.num_neurons
        l = self.num_layers

        ####################################################################################################################

        if arch == 'randomly_connected':
            logic_layers.append(torch.nn.Flatten())
            logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **llkw))
            for _ in range(l - 1):
                logic_layers.append(LogicLayer(in_dim=k, out_dim=k, **llkw))

            model = torch.nn.Sequential(
                *logic_layers,
                GroupSum(class_count, self.tau)
            )

        ####################################################################################################################

        else:
            raise NotImplementedError(arch)

        ####################################################################################################################

        total_num_neurons = sum(map(lambda x: x.num_neurons, logic_layers[1:-1]))
        print(f'total_num_neurons={total_num_neurons}')
        total_num_weights = sum(map(lambda x: x.num_weights, logic_layers[1:-1]))
        print(f'total_num_weights={total_num_weights}')
        # if args.experiment_id is not None:
        #     results.store_results({
        #         'total_num_neurons': total_num_neurons,
        #         'total_num_weights': total_num_weights,
        #     })

        model = model.to('cuda')

        print(model)
        # if args.experiment_id is not None:
        #     results.store_results({'model_str': str(model)})

        loss_fn = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        return model, loss_fn, optimizer

    def _train(self, model, x, y, loss_fn, optimizer):
        x = model(x)
        loss = loss_fn(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def eval(self, model, loader, mode):
        orig_mode = model.training
        with torch.no_grad():
            model.train(mode=mode)
            res = np.mean(
                [
                    (model(batch['features'].to('cuda').round()).argmax(-1) == batch['target'].float().to('cuda')).to(torch.float32).mean().item()
                    for batch in loader
                ]
            )
            model.train(mode=orig_mode)
        return res.item()

    def packbits_eval(self, model, loader):
        orig_mode = model.training
        with torch.no_grad():
            model.eval()
            res = np.mean(
                [
                    (model(PackBitsTensor(batch['features'].to('cuda').reshape(batch['features'].shape[0], -1).round().bool())).argmax(-1) == batch['target'].float().to(
                        'cuda')).to(torch.float32).mean().item()
                    for batch in loader
                ]
            )
            model.train(mode=orig_mode)
        return res.item()

    def train(self, train_loader, validation_loader):
        assert self.num_iterations % self.eval_freq == 0, (
            f'iteration count ({self.num_iterations}) has to be divisible by evaluation frequency ({self.eval_freq})'
        )

        # if args.experiment_id is not None:
        #     assert 520_000 <= args.experiment_id < 530_000, args.experiment_id
        #     results = ResultsJSON(eid=args.experiment_id, path='./results/')
        #     results.store_args(args)

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # train_loader, validation_loader, test_loader = load_dataset(args)
        self.model, loss_fn, optim = self.get_model()

        ####################################################################################################################

        best_acc = 0

        for i, batch in tqdm(
                enumerate(load_n(train_loader, self.num_iterations)),
                desc='iteration',
                total=self.num_iterations,
        ):
            x = batch['features']
            y = batch['target'].float()
            x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[self.training_bit_count]).to('cuda')
            y = y.to('cuda')

            loss = self._train(self.model, x, y, loss_fn, optim)

            if (i + 1) % self.eval_freq == 0:
                if self.extensive_eval:
                    train_accuracy_train_mode = self.eval(self.model, train_loader, mode=True)
                    valid_accuracy_eval_mode = self.eval(self.model, validation_loader, mode=False)
                    valid_accuracy_train_mode = self.eval(self.model, validation_loader, mode=True)
                else:
                    train_accuracy_train_mode = -1
                    valid_accuracy_eval_mode = -1
                    valid_accuracy_train_mode = -1
                train_accuracy_eval_mode = self.eval(self.model, train_loader, mode=False)
                # test_accuracy_eval_mode = eval(model, test_loader, mode=False)
                # test_accuracy_train_mode = eval(model, test_loader, mode=True)

                r = {
                    'train_acc_eval_mode': train_accuracy_eval_mode,
                    'train_acc_train_mode': train_accuracy_train_mode,
                    'valid_acc_eval_mode': valid_accuracy_eval_mode,
                    'valid_acc_train_mode': valid_accuracy_train_mode,
                    # 'test_acc_eval_mode': test_accuracy_eval_mode,
                    # 'test_acc_train_mode': test_accuracy_train_mode,
                }

                if self.use_packbits_eval:
                    r['train_acc_eval'] = self.packbits_eval(self.model, train_loader)
                    r['valid_acc_eval'] = self.packbits_eval(self.model, train_loader)
                    # r['test_acc_eval'] = self.packbits_eval(model, test_loader)

                # if args.experiment_id is not None:
                #     results.store_results(r)
                # else:
                #     print(r)

                if valid_accuracy_eval_mode > best_acc:
                    best_acc = valid_accuracy_eval_mode
                    self.best_model = copy.deepcopy(self.model)
                    # if args.experiment_id is not None:
                    #     results.store_final_results(r)
                    # else:
                    #     print('IS THE BEST UNTIL NOW.')

                # if args.experiment_id is not None:
                #     results.save()

        # early stopping
        self.model = self.best_model

    def predict(self, loader, mode=False):
        orig_mode = self.model.training
        with torch.no_grad():
            self.model.train(mode=mode)
            predictions = torch.cat([self.model(batch['features'].to('cuda')) for batch in loader])
            self.model.train(mode=orig_mode)
        return predictions


# if __name__ == '__main__':

    # ####################################################################################################################
    #
    # if args.compile_model:
    #     print('\n' + '='*80)
    #     print(' Converting the model to C code and compiling it...')
    #     print('='*80)
    #
    #     for opt_level in range(4):
    #
    #         for num_bits in [
    #             # 8,
    #             # 16,
    #             # 32,
    #             64
    #         ]:
    #             os.makedirs('lib', exist_ok=True)
    #             save_lib_path = 'lib/{:08d}_{}.so'.format(
    #                 args.experiment_id if args.experiment_id is not None else 0, num_bits
    #             )
    #
    #             compiled_model = CompiledLogicNet(
    #                 model=model,
    #                 num_bits=num_bits,
    #                 cpu_compiler='gcc',
    #                 # cpu_compiler='clang',
    #                 verbose=True,
    #             )
    #
    #             compiled_model.compile(
    #                 opt_level=1 if args.num_layers * args.num_neurons < 50_000 else 0,
    #                 save_lib_path=save_lib_path,
    #                 verbose=True
    #             )
    #
    #             correct, total = 0, 0
    #             with torch.no_grad():
    #                 for (data, labels) in torch.utils.data.DataLoader(test_loader.dataset, batch_size=int(1e6), shuffle=False):
    #                     data = torch.nn.Flatten()(data).bool().numpy()
    #
    #                     output = compiled_model(data, verbose=True)
    #
    #                     correct += (output.argmax(-1) == labels).float().sum()
    #                     total += output.shape[0]
    #
    #             acc3 = correct / total
    #             print('COMPILED MODEL', num_bits, acc3)
