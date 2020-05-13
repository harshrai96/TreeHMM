#' Initializing treeHMM with given parameters
#'
#' @param States A (2 * 1) vector with first element being discrete state value for the cases(or positive) and second element being discrete state value for the controls(or negative) for given treeHMM
#' @param Symbols List containing (M * 1) vectors for discrete values of emissions(where "M" is the possible number of emissions) for each feature variable
#' @param treemat Adjacent Symmetry Matrix that describes the topology of the tree
#' @param startProbs (N * 1) vector containing starting probabilities for the states, where "N" is the possible number of states(Optional). Default is equally probable states
#' @param transProbs (N * N) matrix containing transition probabilities for the states, where "N" is the possible number of states(Optional)
#' @param emissionProbs List of (N * M) matrices containing emission probabilities for the states, for each feature variable(optional). Default is equally probable emissions
#' @return List describing the parameters of treeHMM(pi, alpha, beta)
#' @examples
#' tmat = matrix(c(0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0),
#'                5,5, byrow= TRUE ) #for "X" (5 nodes) shaped tree
#' states = c("P","N") #"P" represent cases(or positive) and "N" represent controls(or negative)
#' hmmA = initHMM(states,list(c("L","R")), tmat) #one feature with two discrete levels "L" and "R"
#' hmmB = initHMM(states, list(c("X","Y")),tmat, c(0.5,0.5), matrix(c(0.7,0.3,0.3,0.7),2,2))

initHMM = function (States, Symbols, treemat, startProbs = NULL, transProbs = NULL,
                    emissionProbs = NULL)
{
  nStates = length(States)
  nLevel = length(Symbols)
  E=list()
  S = rep((1/nStates),nStates)
  T = 0.5 * diag(nStates) + array(0.5/(nStates), c(nStates,nStates))
  names(S) = States
  dimnames(T) = list(from = States, to = States)
  if (!is.null(startProbs)) {
    S[] = startProbs[]
  }
  if (!is.null(transProbs)) {
    T[, ] = transProbs[, ]
  }
  for(i in 1:nLevel)
  {
    nSymbols=length(Symbols[[i]])
    E[[i]] = array(1/(nSymbols), c(nStates, nSymbols))
    dimnames(E[[i]]) = list(states = States, symbols = Symbols[[i]])
    if (!is.null(emissionProbs[[i]])) {
      E[[i]][, ] = emissionProbs[[i]][, ]
    }
  }
  return(list(States = States, Symbols = Symbols, startProbs = S,
              transProbs = T, emissionProbs = E, adjsym=treemat))
}
