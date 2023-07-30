# 测试5个train_loader是否一致

test_acc1, cm_before1 = utils.test(source_model, device, target_train_loader1, split="test",
                                   num_classes=num_classes)  # confusion matrix's function
per_class_acc_before1 = cm_before1.diagonal().numpy() / cm_before1.sum(axis=1).numpy()
per_class_acc_before1 = per_class_acc_before1.mean() * 100

out_str = '{}->{}, Before {}:\t Avg. acc={:.2f}%\tAgg. acc={:.2f}%'.format(args.source, args.target, args.da_strat,
                                                                           per_class_acc_before1, test_acc1)
print("原始train_loader1: " + out_str)

test_acc2, cm_before2 = utils.test(source_model, device, target_train_loader2, split="test",
                                   num_classes=num_classes)  # confusion matrix's function
per_class_acc_before2 = cm_before2.diagonal().numpy() / cm_before2.sum(axis=1).numpy()
per_class_acc_before2 = per_class_acc_before2.mean() * 100

out_str = '{}->{}, Before {}:\t Avg. acc={:.2f}%\tAgg. acc={:.2f}%'.format(args.source, args.target, args.da_strat,
                                                                           per_class_acc_before2, test_acc2)
print("原始train_loader2: " + out_str)

test_acc3, cm_before3 = utils.test(source_model, device, target_train_loader3, split="test",
                                   num_classes=num_classes)  # confusion matrix's function
per_class_acc_before3 = cm_before3.diagonal().numpy() / cm_before3.sum(axis=1).numpy()
per_class_acc_before3 = per_class_acc_before3.mean() * 100

out_str = '{}->{}, Before {}:\t Avg. acc={:.2f}%\tAgg. acc={:.2f}%'.format(args.source, args.target, args.da_strat,
                                                                           per_class_acc_before3, test_acc3)
print("原始train_loader3: " + out_str)

test_acc4, cm_before4 = utils.test(source_model, device, target_train_loader4, split="test",
                                   num_classes=num_classes)  # confusion matrix's function
per_class_acc_before4 = cm_before4.diagonal().numpy() / cm_before4.sum(axis=1).numpy()
per_class_acc_before4 = per_class_acc_before4.mean() * 100

out_str = '{}->{}, Before {}:\t Avg. acc={:.2f}%\tAgg. acc={:.2f}%'.format(args.source, args.target, args.da_strat,
                                                                           per_class_acc_before4, test_acc4)
print("原始train_loader4: " + out_str)

test_acc5, cm_before5 = utils.test(source_model, device, target_train_loader5, split="test",
                                   num_classes=num_classes)  # confusion matrix's function
per_class_acc_before5 = cm_before5.diagonal().numpy() / cm_before5.sum(axis=1).numpy()
per_class_acc_before5 = per_class_acc_before5.mean() * 100

out_str = '{}->{}, Before {}:\t Avg. acc={:.2f}%\tAgg. acc={:.2f}%'.format(args.source, args.target, args.da_strat,
                                                                           per_class_acc_before5, test_acc5)
print("原始train_loader5: " + out_str)


