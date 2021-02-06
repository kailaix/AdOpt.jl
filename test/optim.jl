import Optim
sess = Session()

@testset "Optim Optimizer" begin 
    x = Variable(zeros(2))
    loss = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2
    for optimizer in [Optim.BFGS(), Optim.LBFGS()]
        opt = OptimOptimizer(optimizer, Optim.Options(iterations=100))
        init(sess)
        losses = Optimize!(sess, loss, opt)
        @test norm(run(sess, x)-[1.0;1.0]) < 1e-5
    end
end