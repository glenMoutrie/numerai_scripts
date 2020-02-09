library(BoomSpikeSlab, quietly = TRUE, warn.conflicts = FALSE, verbose = FALSE)
library(data.table, quietly = TRUE, warn.conflicts = FALSE, verbose = FALSE)

args <- commandArgs(TRUE)

# for (i in args) {
# 	print(i)
# }

tryCatch({
	data <- fread(args[1])
	form <- formula(args[2])

	# Test case
	# data <- birthwt
	# form <- formula("low ~ age  + race + smoke + ptl + ht + ui + ftv")

	mm <- model.matrix(form, data)
	resp <- model.response(model.frame(form, data))

	model <- logit.spike(form, niter = 500, data)
	output <- colMeans(model$beta != 0)

	output <- data.frame(variable = names(output), probability = output)

	write.csv(output, file = args[3], row.names = FALSE)

}, error = function(e) {print(traceback(e)); stop(e)})


