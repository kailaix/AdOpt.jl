@testset "Adaptive BFGS" begin
    sess = Session()
    x = Variable(zeros(2))
    loss = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2
    init(sess)
    opt = AdaptiveBFGSOptimizer(threshold=1e-2)
    losses = Optimize!(sess, loss, opt, 2000)
    @test norm(run(sess, x)-[1.0;1.0]) < 1e-5
end