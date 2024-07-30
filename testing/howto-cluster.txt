# Cluster How-To

## Single Core, X Jobs

1.

ssh jfriedel@sohrab001

2.

source /core/uge/LMP/common/settings.sh

3.

cd $(dir containing qsub*.sh)

4.

qsub -t 1:X -N nop_run -q rostam.q ./qsub_rostam.sh
