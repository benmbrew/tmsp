# Simplest example where ALS is 0.
library(MDPtoolbox)
ALS = 0
B = 87.5
Na = ALS
Nb = floor(B - 1.25*Na)

# define input parameters
Lh = 8.1
Ll = 13.1
mu = 0.75
Rha = 1.25
Rhb = 0.5
Rl = 0.6

# uniformization
mat1 <- matrix(0, nrow = 4, ncol = 4)
mat2 <- matrix(0, nrow = 4, ncol = 4)

# row 1
mat1[1,1] <- 1- ((8.1/60) + (13.1/60))
mat1[1,2] <- ((8.1/60) + (13.1/60))
mat1[1,3] <- 0
mat1[1,4] <- 0

# row 2
mat1[2,1] <- (1*.75)/60
mat1[2,2] <- 1- ((8.1/60) + (13.1/60) + (0.75/60))
mat1[2,3] <- ((8.1/60) + (13.1/60))
mat1[2,4] <- 0

# row 3
mat1[3,1] <- 0
mat1[3,2] <- (2*.75)/60
mat1[3,3] <- 1- ((8.1/60) + (13.1/60) + ((2*0.75)/60))
mat1[3,4] <- ((8.1/60) + (13.1/60))

# row 4
mat1[4,1] <- 0
mat1[4,2] <- 0
mat1[4,3] <- (3*0.75)/60
mat1[4,4] <- 1- ((3*0.75)/60)

# row 1
mat2[1,1] <- 1- ((8.1/60) + (13.1/60))
mat2[1,2] <- ((8.1/60) + (13.1/60))
mat2[1,3] <- 0
mat2[1,4] <- 0

# row 2
mat2[2,1] <- (1*.75)/60
mat2[2,2] <- 1- ((8.1/60) + (13.1/60) + (0.75/60))
mat2[2,3] <- ((8.1/60) + (13.1/60))
mat2[2,4] <- 0

# row 3
mat2[3,1] <- 0
mat2[3,2] <- (2*.75)/60
mat2[3,3] <- 1- ((8.1/60) + (13.1/60) + ((2*0.75)/60))
mat2[3,4] <- ((8.1/60) + (13.1/60))

# row 4
mat2[4,1] <- 0
mat2[4,2] <- 0
mat2[4,3] <- (3*0.75)/60
mat2[4,4] <- 1- ((3*0.75)/60)

# gather the two matrices
P <- array(0, c(4,4,2))
P[,,1] <- mat1
P[,,2] <- mat2

# reward matrix
r_mat <-r_mat <- matrix(0, nrow = 4, ncol = 2)
dimnames(r_mat) <- list(NULL, c('R1', 'R2'))

# row 1
r_mat[1,1] <- 0
r_mat[1,2] <- 0

# row 2
r_mat[2,1] <- Lh*Rhb + Ll*Rl
r_mat[2,2] <- Lh*Rhb + Ll*Rl

# row 3
r_mat[3,1] <- Lh*Rhb + Ll*Rl
r_mat[3,2] <- Lh*Rhb + Ll*Rl

# row 4
r_mat[4,1] <- Lh*Rhb + Ll*Rl
r_mat[4,2] <- Lh*Rhb + Ll*Rl

# run policy iteration
results <- mdp_policy_iteration(P, r_mat, discount = 0.00001)

# Visualise the policy
results$time
results$policy
results$V
mean(round(results$V, 3))

####################################################################################
# Example where ALS = 10
library(MDPtoolbox)
ALS = 5
B = 87.5
Na = ALS
Nb = floor(B - 1.25*Na)

# define input parameters
Lh = 8.1
Ll = 13.1
mu = 0.75
Rha = 1.25
Rhb = 0.5
Rl = 0.6

# uniformization
mat1 <- matrix(0, nrow = 492, ncol = 492)
mat2 <- matrix(0, nrow = 25, ncol = 25)

denom = 60

for(j in 0:Nb){
  for(i in 0:Na){
    if(i == j){
      # first row
      
      mat1[(i+1), (j+1)] <- 1 - (Lh/denom+Ll/denom+(mu*(i+j))/denom)
      
    }
    
  }
}













