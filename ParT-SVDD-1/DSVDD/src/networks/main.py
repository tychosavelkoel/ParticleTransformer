from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .ParT import ParticleTransformer

def build_network(net_name, data_config, **kwargs):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'ParT')
    assert net_name in implemented_networks

    net = None
    
    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()

    if net_name == 'ParT':
        cfg = dict(
            input_dim=len(data_config.input_dicts['pf_features']),
            num_classes=len(data_config.label_value),
            # network configurations\
            pair_input_dim=4,
            use_pre_activation_pair=False,
            embed_dims=[128, 512, 128],
            pair_embed_dims=[64, 64, 64],
            num_heads=8,
            num_layers=8,
            block_params=None,
            #fc_params=[(128,0.1)],
            activation='gelu',
            # misc
            trim=False,
            for_inference=False,
            use_amp=True
        )
        
        cfg.update(**kwargs)
        net = ParticleTransformer(**cfg)
        
    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'ParT')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    if net_name == 'ParT':
        cfg = dict(
            input_dim=len(data_config.input_dicts['pf_features']),
            num_classes=len(data_config.label_value),
            # network configurations\
            pair_input_dim=4,
            use_pre_activation_pair=False,
            embed_dims=[128, 512, 128],
            pair_embed_dims=[64, 64, 64],
            num_heads=8,
            num_layers=8,
            block_params=None,
            fc_params=[(32, 0.1)],
            activation='gelu',
            # misc
            trim=True,
            for_inference=False,
            use_amp=True
        )

        cfg.update(**kwargs)
        aenet = ParticleTransformer_Autoencoder(**cfg)
    return ae_net
