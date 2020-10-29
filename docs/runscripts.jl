using Dates

@info "Running task Sin"
cd("src/assets/Codes/Sin/")
run(`sh run.sh`)
cd(@__DIR__)


@info "Running task Poisson"
cd("src/assets/Codes/Poisson/")
run(`sh run.sh`)
cd(@__DIR__)


@info "Running task LinearElasticity"
cd("src/assets/Codes/LinearElasticity/")
run(`sh run.sh`)
cd(@__DIR__)
