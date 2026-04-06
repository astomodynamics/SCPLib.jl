"""Regression test for control trust-region constraints on continuous problems"""

using Clarabel
using JuMP

if !@isdefined SCPLib
    include(joinpath(@__DIR__, "../src/SCPLib.jl"))
end

mutable struct ControlParams_continuous_trust_region_u
    u::Vector{Float64}
    function ControlParams_continuous_trust_region_u()
        new(zeros(1))
    end
end

function test_continuous_trust_region_u()
    nx = 1
    nu = 1
    N = 3

    params = ControlParams_continuous_trust_region_u()
    times = LinRange(0.0, 1.0, N)
    x_ref = zeros(nx, N)
    u_ref = zeros(nu, N - 1)

    eom! = function (dx, x, p, t)
        dx[1] = p.u[1]
        return
    end

    objective = function (x, u)
        return sum(u.^2)
    end

    prob = SCPLib.ContinuousProblem(
        Clarabel.Optimizer,
        eom!,
        params,
        objective,
        times,
        x_ref,
        u_ref,
    )
    set_silent(prob.model)

    @constraint(prob.model, prob.model[:x][1,1] == 0.0)
    @constraint(prob.model, prob.model[:x][1,end] == 0.0)

    algo = SCPLib.SCvxStar(nx, N; w0 = 1.0, nu = nu, use_trustregion_control = true)
    SCPLib.set_trust_region_constraints!(algo, prob, x_ref, u_ref)

    @test length(prob.model[:constraint_trust_region_u_lb]) == N - 1
    @test length(prob.model[:constraint_trust_region_u_ub]) == N - 1
end

test_continuous_trust_region_u()
