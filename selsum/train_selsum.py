#!/usr/bin/env python3 -u
# Trains the SelSum model where the posterior and summarizer are
# trained/optimized separately

from selsum.fairseq_cli_mod.selsum_train import cli_main


if __name__ == '__main__':
    cli_main()
