function cyclic_LR(epoch, total_epochs; swag = false, lr_init=0.01, swag_lr=1e-3, swag_start = 2000)
	t = swag ? ((epoch + 1) / swag_start) : ((epoch + 1) / total_epochs)
	lr_ratio = swag ? (swag_lr / lr_init) : 0.05
	if t <= 0.5
	    factor = 1.0
	elseif t <= 0.9
	    factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
	else
	    factor = lr_ratio
	end

	return (factor * lr_init)
end

function my_train(epoches, L, ps, data, opt; lr_init = 1e-2, swag_start = epoches, cyclic_lr = false)
	local training_loss
	for ep in 1:epoches
		for d in data
			gs = gradient(ps) do
				training_loss = L(d...)
				return training_loss
		    end

		    Flux.update!(opt, ps, gs)
		end
		if cyclic_lr
			opt.eta =  cyclic_LR(ep, epoches, lr_init=lr_init, swag_start = swag_start)
		end
		@show opt.eta ep training_loss
	end
end