# Experiment Models
### Exp 1

```julia
using Pumas, MCMCChains, StatsPlots, GibbsSampler, Distributions

theopmodel_bayes = @model begin
  @param begin

    # Mode at diagm(fill(0.2, 3))
    Ω1 ~ Uniform(0.1,0.6)
    Ω2 ~ Uniform(0.1,0.6)
    Ω3 ~ Uniform(0.1,0.6)

    # Mean at 0.5 and positive density at 0.0
    σ ~ Gamma(1.0, 0.5)
  end

  @random begin
    η1 ~ Normal(Ω1)
    η2 ~ Normal(Ω2)
    η3 ~ Normal(Ω3)
  end

  @pre begin                *exp(η3)
    Ka = (SEX == 1 ? 0.3 : 0.4)*exp(η1)
    CL = 0.2*(WT/70)            *exp(η2)
    Vc = 0.5                    *exp(η3)
  end

  @covariates SEX WT

  @dynamics Depots1Central1

  @derived begin
    # The conditional mean
    μ := @. Central / Vc
    # Additive error model
    dv ~ @. Normal(μ, σ)
  end
end

data = read_pumas(example_data("event_data/THEOPP"), covariates = [:SEX,:WT])

param = init_param(theopmodel_bayes)
result = fit(theopmodel_bayes, data, param, Pumas.BayesGibbs(MH(n_samples = 35));nsamples=1000)
```

#### Gibbs with MH, n_samples = 35
```

Time elapsed: 1:08
parameters      mean       std   naive_se      mcse        ess      rhat

Ω1[1]    0.3547    0.1443     0.0036    0.0062   562.0578    1.0043
Ω2[1]    0.3598    0.1411     0.0035    0.0052   638.6811    0.9994
Ω3[1]    0.3597    0.1416     0.0035    0.0050   582.7379    0.9994
σ[1]    0.8263    0.0778     0.0019    0.0147     5.5371    1.5077

```

#### BayesMCMC
```julia
result = fit(theopmodel_bayes, data, param, Pumas.BayesMCMC();nsamples=2000, nadapts = 1000)
chains = Pumas.Chains(result)
```
```
Time elapsed: 00:41
parameters      mean       std   naive_se      mcse        ess      rhat

    Ω1    0.5130    0.0805     0.0025    0.0045   367.3842    1.0061
    Ω2    0.1493    0.0486     0.0015    0.0025   374.7807    0.9998
    Ω3    0.2336    0.1098     0.0035    0.0064   152.7344    1.0051
     σ    0.7224    0.1433     0.0045    0.0091   222.0662    1.0021
```


Trace Plot

![Bayes MCMC](img/traceplot_bayesmcmc.png)
![Gibbs](img/traceplot.png)

Histogram Plot

![Bayes MCMC](img/histogram_bayesmcmc.png)
![Gibbs](img/histogram.png)

Autocorrelation Plot

![Bayes MCMC](img/autocorr_bayesmcmc.png)
![Gibbs](img/autocor.png)