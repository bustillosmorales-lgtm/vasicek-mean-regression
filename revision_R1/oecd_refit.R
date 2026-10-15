suppressMessages({library(gamlss); library(vasicekreg)})
args <- commandArgs(TRUE)
d <- read.csv(args[1])
f <- educ ~ homicide + dwellings + labour_insec
mb <- gamlss(f, family = BE(), data = d, trace = FALSE)
mv <- gamlss(f, family = NVASIM(), data = d, trace = FALSE)
rep1 <- function(m, nm) {
  s <- summary(m, save = TRUE)
  est <- s$coef.table[1:4, 1]; se <- s$coef.table[1:4, 2]
  z <- est / se
  data.frame(model = nm, par = names(est), est = est, se = se,
             p_t = s$coef.table[1:4, 4], p_norm = 2 * pnorm(-abs(z)),
             df_t = m$df.residual)
}
o <- rbind(rep1(mb, "beta"), rep1(mv, "vasicek"))
print(o, digits = 6)
cat("loglik beta", logLik(mb), " vasicek", logLik(mv), "\n")
cat("beta sigma(gamlss) ", fitted(mb, "sigma")[1], " -> phi ", 1 / fitted(mb, "sigma")[1]^2 - 1, "\n")
cat("vasicek theta ", fitted(mv, "sigma")[1], "\n")
write.csv(data.frame(country = d$country, y = d$educ,
                     mu_beta = fitted(mb, "mu"), mu_vas = fitted(mv, "mu"),
                     rq_beta = resid(mb), rq_vas = resid(mv),
                     sigma_beta = fitted(mb, "sigma"), theta_vas = fitted(mv, "sigma")),
          args[2], row.names = FALSE)
