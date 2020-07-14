import argparse

import rsabias.core.tasks as tasks


def main():
    parser = argparse.ArgumentParser(description='RSA keys bias analysis')
    actions = ['convert', 'dist', 'plot', 'dist+plot', 'group', 'split', 'marginal', 'filter', 'evaluate', 'build', 'batch_gcd', 'visualize', 'classify']
    methods = ['complex', 'naive']
    formats = ['json', 'csv']
    parser.add_argument('-a', '--action', type=str, choices=actions,
                        required=True)
    parser.add_argument('-t', '--trans', type=str, required=False)
    parser.add_argument('-i', '--inp', type=str, required=True)
    parser.add_argument('-o', '--out', type=str, required=True)
    parser.add_argument('-f', '--format', type=str, choices=formats,
                        default='json')
    parser.add_argument('-g', '--groups', type=str, default=None)
    parser.add_argument('-d', '--decompress', action='store_true',
                        default=False)
    parser.add_argument('-s', '--subspaces', type=str)
    parser.add_argument('-c', '--classtable', type=str, required=False, default=None)
    parser.add_argument('-m', '--method', type=str, required=False, choices=methods, default='complex')
    parser.add_argument('-l', '--labels', action='store_true', default=False)
    parser.add_argument('-p', '--prime_wise', action='store_true', default=False)
    parser.add_argument('-r', '--remove_duplicities', action='store_true', default=False)

    args = parser.parse_args()

    if args.action == 'convert':
        tasks.convert(args.inp, args.out, args.trans, count_dist=True, compress=not args.decompress,
                      distributions_only=False, out_format=args.format, prime_wise=args.prime_wise,
                      remove_duplicities=args.remove_duplicities)
    elif args.action == 'split':
        tasks.split(args.inp, args.out, n_test_keys=10000, out_format=args.format, compress=not args.decompress)
    elif args.action == 'dist':
        tasks.distributions(args.inp, args.out, args.trans, prime_wise=args.prime_wise)
    elif args.action == 'plot':
        tasks.plot_dist(args.inp, args.out, args.trans)
    elif args.action == 'dist+plot':
        tasks.distributions(args.inp, args.out, args.trans, draw_plot=True, prime_wise=args.prime_wise)
    elif args.action == 'group':
        tasks.cluster(args.inp, args.out, args.trans)
    elif args.action == 'marginal':
        tasks.marginal(args.inp, args.out, args.trans, args.subspaces)
    elif args.action == 'filter':
        tasks.filter_test_dataset(args.inp, args.out, args.groups, out_format=args.format)
    elif args.action == 'evaluate':
        tasks.evaluate(args.inp, args.out, args.trans, args.groups, args.method, args.classtable, args.labels)
    elif args.action == 'build':
        tasks.build_classification_table(args.trans, args.groups, args.inp, args.out, args.method)
    elif args.action == 'batch_gcd':
        tasks.batch_gcd(args.inp, args.out, args.trans, args.groups, args.method, args.classtable)
    elif args.action == 'visualize':
        tasks.visualize_model_performance(args.inp, args.out)
    elif args.action == 'classify':
        tasks.classify_key(args.inp, args.out)


if __name__ == '__main__':
    main()
