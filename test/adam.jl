sess = Session()

@testset "ADAM" begin 
    x = Variable(zeros(2))
    loss = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2
    init(sess)
    losses = Optimize!(sess, loss, ADAMOptimizer())
    @test norm(run(sess, x)-[1.0;1.0]) < 1e-5
end