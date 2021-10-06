# this script will explore methods in R for Markov decision processes

# resources: 
# https://rpubs.com/relund/MDP2#:~:text=The%20MDP2%20package%20in%20R%20is%20a%20package%20for%20solving,%2DMDPs%20(HMDPs)%20(A.%20R. 
# https://miat.inrae.fr/MDPtoolbox/QuickStart.pdf
# https://cran.r-project.org/web/packages/MDPtoolbox/MDPtoolbox.pdf
# https://onlinelibrary.wiley.com/doi/full/10.1111/ecog.00888
library(MDPtoolbox)
# figure out what each element is that's generated above
# A = actions, in this case 2, so P consists of 2 matrices of S dimensions
# S = States, number of states 
# r1 = the reward when forest is in oldest state and action "wait" (1) is performed
# r2 = the reward when forest is in oldest state and action "cut" (2) is peformed.

# Generates a MDP for a simple forest management problem
MDP <- mdp_example_forest()

# Find an optimal policy
results <- mdp_policy_iteration(MDP$P, MDP$R, 0.9)

# Visualise the policy
results$time

########
# try to reproduce the chong vehicle mix results 
########

# B = total budget
# Total ALS vehicles is Na
# total BLS vehicles is Nb
# Busy ALS vehicles is i
# Busy BLS vehicles is j 
# mu is service time rate (3/4)
# lambdaH is High priority calls per hour = 8.1
# lambdaL is low priority calls per hour = 13.1
# Rha = reward for dispatching ALS unit to high priority cse =1
# Rhb = reward for dispatching BLS unit to high priority case = 0.5
# Rl = reward for dispatching ALS or BLS to low priority case = 0.6
# Ca = Annual operating cost for ALS = 650k (normalized to 1.25)
# Cb = Annual operating cost for BLS = 520k (normalized to 1)

# start with:
# Na = 70, B = 87.5
# Evaluate vehicle mixes in the set: Na <= 70 & Nb = 87.5 - 1.25*Na
# so scenario 1: Na = 0, Nb = 87.5
# scenario 2: Na = 1, Nb = 87.5 - 1.25(1) = 86.25
# scenario 10: Na =10, Nb = 87.5 - 1.25(10) = 75

# we have 70 scenarios, each scenario has a finite number of states. 

# start in scenario where Na = 0, all the way to Na = 70, save the long run average reward 

# create a function for uniformizing probabilities 
# lambdaH+LambdaL+(i +j)*u = X
# then divide each term by X to get probabilities
library(MDPtoolbox)
ALS =10
B = 87.5
Na = ALS
Nb = floor(B - 1.25*Na)

mat1 <- matrix(0, nrow = 4, ncol = 4)
mat2 <- matrix(0, nrow = 4, ncol = 4)

i = 3
j = 1
for(i in 1:4){
  for (j in 1:4){
    print(paste0('row ', i, ' and column ', j))
    # condition for the diagnol
    denom = 8.1+13.1+i*(.75)+j*(.75)
    lambdah <- 8.1/denom
    lambdal <- 13.1/denom
    adj_j <- j*(.75)/denom
    adj_i <- i*(.75)/denom
    if(i == j){
     
      mat1[i,j] <- 1-lambdah-lambdal-adj_j-adj_i
      mat2[i,j] <- 1-lambdah-lambdal-adj_j
      
      # condtion for unit service completion of i
    } else if(j == i - 1){
      mat1[i,j] <- adj_i
      mat2[i,j] <- adj_j
      
      # condtion for arrival of high priority 
    } else if (j == i + 1){
      mat1[i,j] <- lambdal+lambdah # might need to uniformize this
      mat2[i,j] <- lambdal+lambdah # might need to uniformize this
    } 
  }
}
P <- array(0, c(88,88,2))
P[,,1] <- mat1
P[,,2] <- mat2
# reward matrix 
# S X A
r_mat <- matrix(0, nrow = 88, ncol = 2)
# action 1 always gets you lambdah*Rha because no BLS present
r_mat[,1] <- (8.1*0.5)+(13.1*0.6)
# action 2 always gets you reward from action 1 plus lambdal*Rl
r_mat[,2] <- (8.1*0.5)+(13.1*0.6)
dimnames(r_mat) <- list(NULL, c('R1', 'R2'))

# run policy iteration
results <- mdp_policy_iteration(P, r_mat, discount = 0.00001)

# Visualise the policy
results$time
results$policy
results$V
mean(results$V)

############################################################
library(MDPtoolbox)
# manually create matrix 
ALS = 10
B = 87.5
Na = ALS
Nb = floor(B - 1.25*Na)

mat1 <- matrix(0, nrow = 4, ncol = 4)
mat2 <- matrix(0, nrow = 4, ncol = 4)

(9*.75)/(8.1+13.1+9*(.75)+74*(.75))
(74*.75)/(8.1+13.1+9*(.75)+74*(.75))

# first row mat1
mat1[1,1] <- 0
mat1[1,2] <- 8.1/(8.1+13.1+9*(.75)+74*(.75))
mat1[1,3] <- 13.1/(8.1+13.1+9*(.75)+74*(.75))
mat1[1,4] <- 0

# second row mat1
mat1[2,1] <- (9*(.75))/(8.1+13.1+10*(.75)+74*(.75))
denom <- (8.1+13.1+10*(.75)+74*(.75))
lambdah <- 8.1/denom
lambdal <- 13.1/denom
adj_i <- (10*.75)/denom
adj_j <- (74*.75)/denom
mat1[2,2] <- 1 - lambdal-adj_i-adj_j
mat1[2,3] <- 0
denom <- (8.1+13.1+10*(.75)+74*(.75))
lambdal <- 13.1/denom
mat1[2,4] <- lambdal

# third row mat1
mat1[3,1] <- (75*.75)/(8.1+13.1+9*(.75)+75*(.75))
mat1[3,2] <- 0
denom <- (8.1+13.1+9*(.75)+75*(.75))
lambdal <- 13.1/denom
lambdah <- 8.1/denom
adj_i <- (9*.75)/denom
adj_j <- (75*.75)/denom
mat1[3,3] <- 1 - lambdah-adj_i-adj_j
mat1[3,4] <- lambdah

# fourth row mat1
mat1[4,1] <- 0
mat1[4,2] <- (75*.75)/(8.1+13.1+10*(.75)+75*(.75))
mat1[4,3] <- (10*(.75))/(8.1+13.1+10*(.75)+75*(.75))
denom <- (8.1+13.1+10*(.75)+75*(.75))
adj_i <- (10*.75)/denom
adj_j <- (75*.75)/denom
mat1[4,4] <- 1 - adj_j - adj_i

# create mat2 
mat2 <- mat1
denom <- (8.1+13.1+9*(.75)+75*(.75))
lambdal <- 13.1/denom
lambdah <- 8.1/denom
mat2[3,4] <- lambdah + lambdal

# gather the two matrices
P <- array(0, c(4,4,2))
P[,,1] <- mat1
P[,,2] <- mat2

# reward matrix
r_mat <-r_mat <- matrix(0, nrow = 4, ncol = 2)
dimnames(r_mat) <- list(NULL, c('R1', 'R2'))

# state (row) 1
r_mat[1,1] <- 8.1*1 + 13.1*.6
r_mat[1,2] <- 8.1*1 + 13.1*.6

# state (row) 2
r_mat[2,1] <- 8.1*.5 + 13.1*.6
r_mat[2,2] <- 8.1*.5 + 13.1*.6

# state (row) 3
r_mat[3,1] <- 8.1*1
r_mat[3,2] <- 8.1*1 + 13.1*.6

# state (row) 4
r_mat[4,1] <- 0
r_mat[4,2] <- 0

# run policy iteration
results <- mdp_policy_iteration(P, r_mat, discount = 0.00001)

# Visualise the policy
results$time
results$policy
results$V
mean(results$V)

# mdp_policy_iteration <- function (P, R, discount, policy0, max_iter, eval_type) {
#   start <- as.POSIXlt(Sys.time())
#   if (discount <= 0 | discount > 1) {
#     print("--------------------------------------------------------")
#     print("MDP Toolbox ERROR: Discount rate must be in ]0; 1]")
#     print("--------------------------------------------------------")
#   }
#   else if (nargs() > 3 & is.list(P) & ifelse(!missing(policy0), 
#                                              length(policy0) != dim(P[[1]])[1], F)) {
#     print("--------------------------------------------------------")
#     print("MDP Toolbox ERROR: policy must have the same dimension as P")
#     print("--------------------------------------------------------")
#   }
#   else if (nargs() > 3 & !is.list(P) & ifelse(!missing(policy0), 
#                                               length(policy0) != dim(P)[1], F)) {
#     print("--------------------------------------------------------")
#     print("MDP Toolbox ERROR: policy must have the same dimension as P")
#     print("--------------------------------------------------------")
#   }
#   else if (nargs() > 4 & ifelse(!missing(max_iter), max_iter <= 
#                                 0, F)) {
#     print("--------------------------------------------------------")
#     print("MDP Toolbox ERROR: The maximum number of iteration must be upper than 0")
#     print("--------------------------------------------------------")
#   }
#   else {
#     if (is.list(P)) {
#       S <- dim(P[[1]])[1]
#       A <- length(P)
#     }
#     else {
#       S <- dim(P)[1]
#       A <- dim(P)[3]
#     }
#     PR <- mdp_computePR(P, R)
#     if (nargs() < 6) {
#       eval_type <- 0
#     }
#     if (nargs() < 5) {
#       max_iter <- 1000
#     }
#     if (nargs() < 4) {
#       bellman <- mdp_bellman_operator(P, PR, discount, 
#                                       numeric(S))
#       Vunused <- bellman[[1]]
#       policy0 <- bellman[[2]]
#     }
#     iter <- 0
#     policy <- policy0
#     is_done <- F
#     while (!is_done) {
#       iter <- iter + 1
#       if (eval_type == 0) {
#         V <- mdp_eval_policy_matrix(P, PR, discount, 
#                                     policy)
#       }
#       else {
#         V <- mdp_eval_policy_iterative(P, PR, discount, 
#                                        policy)
#       }
#       bellman <- mdp_bellman_operator(P, PR, discount, 
#                                       V)
#       Vnext <- bellman[[1]]
#       policy_next <- bellman[[2]]
#       n_different <- sum(policy_next != policy)
#       if (setequal(policy_next, policy) | iter == max_iter) {
#         is_done <- T
#       }
#       else {
#         policy <- policy_next
#       }
#     }
#     end <- as.POSIXlt(Sys.time())
#     return(list(V = V, policy = policy, iter = iter, time = end - 
#                   start))
#   }
# }
