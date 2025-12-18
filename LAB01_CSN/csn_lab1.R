library(igraph)

# PLOT (a)

# We chose to compute the calculation of the C(p) and the L(p) values 20 times
# for every value of p that we used, and then we calculated the mean between
# them. We iterated over twelve values of p.

iter=20
WS <- sample_smallworld(dim=1,size=1200, nei=4,p=0)
seq_p <- 10^seq(log10(0.0001), log10(1), length.out = 12)

# Created empty lists for the C(p) values and the L(p) values

C_ratio <- numeric(length(seq_p))
L_ratio <- numeric(length(seq_p))

C_0 <- transitivity(WS)
L_0 <- mean_distance(WS)

# Iteration for every value of p

for(i in 1:length(seq_p)) {
  c_vector=c()
  l_vector=c()
  for (j in 1:iter) {
    WS <- sample_smallworld(dim = 1, size = 1200, nei = 4, p = seq_p[i])
    
    C_p <- transitivity(WS, type = "global")
    L_p <- mean_distance(WS)
    
    c_vector <- c(c_vector, C_p)
    l_vector <- c(l_vector, L_p)
  }
  
  C_ratio[i] <- mean(c_vector) / C_0
  L_ratio[i] <- mean(l_vector) / L_0
}

# Plot of the points we generated

plot(seq_p, C_ratio, 
     log = "x",                  
     xlim = c(min(seq_p), max(seq_p)),
     ylim = c(0, 1),              
     pch = 22,                  
     bg = "white",        
     col = "black",             
     cex = 1.2,                 
     xlab = "p",                
     ylab = "",                   
     main = "",                 
     cex.lab = 1.2,               
     las = 1)   

points(seq_p, L_ratio, 
       pch = 19,                
       col = "black", 
       cex = 1.2)

text(0.2, 0.85, expression(italic(C(p)/C(0))), cex = 1.2)  
text(0.0003, 0.2, expression(italic(L(p)/L(0))), cex = 1.2) 

grid()

# PLOT (b)

# Generated the sequence of n
seq_n <- 2^(3:16)

list <- numeric(length(seq_n))
iter <- 10
epsilon <- 0.1
# Executed this for cycle

for (idx in seq_along(seq_n)) {
  n <- seq_n[idx]
  p <- (1+epsilon)*log(n)/ n
  distances <- numeric(iter)  
  
  for (r in 1:iter) {
    gnp <- sample_gnp(n, p)
    distances=c(distances,mean_distance(gnp))
  }
  list[idx] <- mean(distances, na.rm = TRUE)
}

# Plotted the points
plot(seq_n, list, type = "b", xlab = "Network size (n)", ylab = "Average shortest path length", main ='Average Shortest-Path length for the Erdos-Rényi model')
grid()
