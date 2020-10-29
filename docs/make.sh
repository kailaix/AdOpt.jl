cd src/assets/Codes/Sin/
sh run.sh 
cd "$(dirname "$0")"

cd src/assets/Poisson/
sh run.sh 
cd "$(dirname "$0")"

git add -f src/assets/Codes/Sin/sinloss*.png
git add -f src/assets/Codes/Poisson/*.png
git commit -m "collect results"
git push