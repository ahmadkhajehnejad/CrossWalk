nodes=500
Pred=0.6
Pblue=0.25

Prr=0.025
Pbb=0.025
Pgg=0.025

Pact=0.03

for PrbPrg in '0.001_0.0005'; do # '0.001_0.0005' '0.005_0.0025' '0.01_0.005' '0.015_0.0075' '0.02_0.01' '0.025_0.0125'; do
	Prb=${PrbPrg%_*}
	Prg=${PrbPrg#*_}
	Pbg=$Prg
	python fairinfMaximization.py --nodes $nodes --Pred $Pred --Pblue $Pblue --Prr $Prr --Pbb $Pbb --Pgg $Pgg --Prb $Prb --Prg $Prg --Pbg $Pbg  --Pact $Pact
done



#nodes=500
#Pred=0.7
#
#Phom=0.025
#Pact=0.03
#
#for Phet in 0.001; do #0.001 0.005 0.01 0.015 0.02 0.025; do
#	echo '      '
#	echo '              farinfMaximization.py              Phet='$Phet
#	python fairinfMaximization.py --nodes 500 --Pred $Pred --Phom $Phom --Phet $Phet --Pact $Pact
#done

