import matplotlib.pyplot as plt


def accuracy_vs_epoch(accuracy, epoch, path, type, label=[]):
    if len(label) > 0:
        for i in range(0, len(accuracy)):
            plt.scatter(epoch, accuracy[i], label=label[i])
    else:
        for i in range(0, len(accuracy)):
            plt.scatter(epoch, accuracy[i])
    plt.xlabel("epochs")
    plt.ylabel(type+" accuracy")
    plt.title(type+" accuracy vs epochs")
    if len(label) > 0:
        plt.legend()
    plt.savefig(path)
    plt.close()

def accuracy_vs_learning_rate(accuracy, learning_rate, path, type, label=[]):
    if len(label) > 0:
        for i in range(0, len(accuracy)):
            plt.scatter(learning_rate, accuracy[i], label=label[i])
    else:
        for i in range(0, len(accuracy)):
            plt.scatter(learning_rate, accuracy[i])
    plt.xlabel("learning rate")
    plt.ylabel(type+" accuracy")
    plt.title(type+" accuracy vs learning rate")
    if len(label) > 0:
        plt.legend()
    plt.savefig(path)
    plt.close()

def accuracy_vs_learning_rate_epoch_all(accuracy, learning_rate, path, type, label):
    acc = accuracy[0]
    rate = learning_rate
    rate.append(learning_rate)
    acc.append(accuracy[1])
    for i in range(0, len(label)):
        plt.scatter(rate[i], acc[i], label=str(label[i])+" epochs")
    plt.xlabel("learning rate")
    plt.ylabel(type + " accuracy")
    plt.title(type + " accuracy vs learning rate")
    if len(label) > 0:
        plt.legend()
    plt.savefig(path)
    plt.close()

def accuracy_vs_learning_rate_epoch(accuracy, learning_rate, path, type, label):
    print("acc ",accuracy)
    print("learn ",learning_rate)
    for i in range(0, len(label)):
        print("THIS IS THE COMPARE", learning_rate[i], "NOW ACCURACY ", accuracy[i])
        plt.scatter(learning_rate[i], accuracy[i], label=str(label[i])+" epochs")
        print("work")
    plt.xlabel("learning rate")
    plt.ylabel(type + " accuracy")
    plt.title(type + " accuracy vs learning rate")
    plt.legend()
    plt.savefig(path)
    plt.close()

def layer_count_vs_err(layer_count, err, path, error_label):
    plt.scatter(layer_count, err)
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel(error_label)
    plt.title(error_label+" vs Number of Hidden Layers")
    plt.savefig(path)
    plt.close()

def layer_count_vs_all_same_err(layer_count, mse, bin, path, type):
    plt.scatter(layer_count, mse, color="orange", label="MSE")
    plt.scatter(layer_count, bin, color="green", label="Binary Crossentropy")
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel(type+" Error values")
    plt.title(type+" Error Values vs Number of Hidden Layers")
    plt.legend()
    plt.savefig(path)
    plt.close()

def layer_count_vs_all_err(layer_count, mse_test, bin_test, mse_train, bin_train, path):
    plt.scatter(layer_count, mse_test, color="blue", label="MSE Test")
    plt.scatter(layer_count, mse_train, color="green", label="MSE Train")
    plt.scatter(layer_count, bin_test, color="orange", label="Crossentropy Test")
    plt.scatter(layer_count, bin_train, color="purple", label="Crossentropy Train")
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Error Values")
    plt.title("Num. Hidden Layers vs Test and Train Error")
    plt.legend()
    plt.savefig(path)
    plt.close()

#Really Specific Function for this project to call to generate Box graphs to count instances.
#Only call to get data Once. Not relavent to the current data set.
def BoxPlotting(df):
    width = .35

    X1 = df.processed_data[df.processed_data['Class'] != 2]['Uniformity_of_Cell_Size']
    x_axis = [1,2,3,4,5,6,7,8,9,10]
    y_axis = [len(X1[X1 == 1]),len(X1[X1 == 2]),len(X1[X1 == 3]),
              len(X1[X1 == 4]),len(X1[X1 == 5]),len(X1[X1 == 6]),
              len(X1[X1 == 7]),len(X1[X1 == 8]),len(X1[X1 == 9]),
              len(X1[X1 == 10])]
    plt.bar(x_axis,y_axis,width, color='orange', label="Malignant")
    X1 = df.processed_data[df.processed_data['Class'] != 4]['Uniformity_of_Cell_Size']
    y_axis = [len(X1[X1 == 1]), len(X1[X1 == 2]), len(X1[X1 == 3]),
              len(X1[X1 == 4]), len(X1[X1 == 5]), len(X1[X1 == 6]),
              len(X1[X1 == 7]), len(X1[X1 == 8]), len(X1[X1 == 9]),
              len(X1[X1 == 10])]
    plt.bar([x+width for x in x_axis], y_axis, width, color='green', label="Benign")
    plt.xlabel('Uniformity_of_Cell_Size')
    plt.ylabel('Instance Count of Class')
    plt.legend(loc="best")
    plt.title('Instance Count of Class versus Uniformity_of_Cell_Size')
    plt.savefig('Graph/uniformity_cell_size')
    plt.close()

    X2 = df.processed_data[df.processed_data['Class'] != 2]['Uniformity_of_Cell_Shape']
    y_axis = [len(X2[X2 == 1]), len(X2[X2 == 2]), len(X2[X2 == 3]),
              len(X2[X2 == 4]), len(X2[X2 == 5]), len(X2[X2 == 6]),
              len(X2[X2 == 7]), len(X2[X2 == 8]), len(X2[X2 == 9]),
              len(X2[X2 == 10])]
    plt.bar(x_axis, y_axis, width, color='orange', label="Malignant")
    X2 = df.processed_data[df.processed_data['Class'] != 4]['Uniformity_of_Cell_Shape']
    print(X2)
    y_axis = [len(X2[X2 == 1]), len(X2[X2 == 2]), len(X2[X2 == 3]),
              len(X2[X2 == 4]), len(X2[X2 == 5]), len(X2[X2 == 6]),
              len(X2[X2 == 7]), len(X2[X2 == 8]), len(X2[X2 == 9]),
              len(X2[X2 == 10])]
    plt.bar([x + width for x in x_axis], y_axis, width, color='green', label="Benign")
    plt.xlabel('Uniformity_of_Cell_Shape')
    plt.ylabel('Instance Count of Class')
    plt.legend(loc="best")
    plt.title('Instance Count of Class versus Uniformity_of_Cell_Shape')
    plt.savefig('Graph/uniformity_cell_shape')
    plt.close()

    X3 = df.processed_data[df.processed_data['Class'] != 2]['Bland_Chromatin']
    y_axis = [len(X3[X3 == 1]), len(X3[X3 == 2]), len(X3[X3 == 3]),
              len(X3[X3 == 4]), len(X3[X3 == 5]), len(X3[X3 == 6]),
              len(X3[X3 == 7]), len(X3[X3 == 8]), len(X3[X3 == 9]),
              len(X3[X3 == 10])]
    plt.bar(x_axis, y_axis, width, color='orange', label="Malignant")
    X3 = df.processed_data[df.processed_data['Class'] != 4]['Bland_Chromatin']
    y_axis = [len(X3[X3 == 1]), len(X3[X3 == 2]), len(X3[X3 == 3]),
              len(X3[X3 == 4]), len(X3[X3 == 5]), len(X3[X3 == 6]),
              len(X3[X3 == 7]), len(X3[X3 == 8]), len(X3[X3 == 9]),
              len(X3[X3 == 10])]
    plt.bar([x + width for x in x_axis], y_axis, width, color='green', label="Benign")
    plt.xlabel('Bland_Chromatin')
    plt.ylabel('Instance Count of Class')
    plt.legend(loc="best")
    plt.title('Instance Count of Class versus Bland_Chromatin')
    plt.savefig('Graph/bland_chromatin')
    plt.close()

    X4 = df.processed_data[df.processed_data['Class'] != 2]['Clump_Thickness']
    y_axis = [len(X4[X4 == 1]), len(X4[X4 == 2]), len(X4[X4 == 3]),
              len(X4[X4 == 4]), len(X4[X4 == 5]), len(X4[X4 == 6]),
              len(X4[X4 == 7]), len(X4[X4 == 8]), len(X4[X4 == 9]),
              len(X4[X4 == 10])]
    plt.bar(x_axis, y_axis, width, color='orange', label="Malignant")
    X4 = df.processed_data[df.processed_data['Class'] != 4]['Clump_Thickness']
    y_axis = [len(X4[X4 == 1]), len(X4[X4 == 2]), len(X4[X4 == 3]),
              len(X4[X4 == 4]), len(X4[X4 == 5]), len(X4[X4 == 6]),
              len(X4[X4 == 7]), len(X4[X4 == 8]), len(X4[X4 == 9]),
              len(X4[X4 == 10])]
    plt.bar([x + width for x in x_axis], y_axis, width, color='green', label="Benign")
    plt.xlabel('Clump_Thickness')
    plt.ylabel('Instance Count of Class')
    plt.legend(loc="best")
    plt.title('Instance Count of Class versus Clump_Thickness')
    plt.savefig('Graph/clump_thickness')
    plt.close()

    X5 = df.processed_data[df.processed_data['Class'] != 2]['Marginal_Adhesion']
    y_axis = [len(X5[X5 == 1]), len(X5[X5 == 2]), len(X5[X5 == 3]),
              len(X5[X5 == 4]), len(X5[X5 == 5]), len(X5[X5 == 6]),
              len(X5[X5 == 7]), len(X5[X5 == 8]), len(X5[X5 == 9]),
              len(X5[X5 == 10])]
    plt.bar(x_axis, y_axis, width, color='orange', label="Malignant")
    X5 = df.processed_data[df.processed_data['Class'] != 4]['Marginal_Adhesion']
    y_axis = [len(X5[X5 == 1]), len(X5[X5 == 2]), len(X5[X5 == 3]),
              len(X5[X5 == 4]), len(X5[X5 == 5]), len(X5[X5 == 6]),
              len(X5[X5 == 7]), len(X5[X5 == 8]), len(X5[X5 == 9]),
              len(X5[X5 == 10])]
    plt.bar([x + width for x in x_axis], y_axis, width, color='green', label="Benign")
    plt.xlabel('Marginal_Adhesion')
    plt.ylabel('Instance Count of Class')
    plt.legend(loc="best")
    plt.title('Instance Count of Class versus Marginal_Adhesion')
    plt.savefig('Graph/marginal_adhesion')
    plt.close()

    X6 = df.processed_data[df.processed_data['Class'] != 2]['Single_Epithelial_Cell_Size']
    y_axis = [len(X6[X6 == 1]), len(X6[X6 == 2]), len(X6[X6 == 3]),
              len(X6[X6 == 4]), len(X6[X6 == 5]), len(X6[X6 == 6]),
              len(X6[X6 == 7]), len(X6[X6 == 8]), len(X6[X6 == 9]),
              len(X6[X6 == 10])]
    plt.bar(x_axis, y_axis, width, color='orange', label="Malignant")
    X6 = df.processed_data[df.processed_data['Class'] != 4]['Single_Epithelial_Cell_Size']
    y_axis = [len(X6[X6 == 1]), len(X6[X6 == 2]), len(X6[X6 == 3]),
              len(X6[X6 == 4]), len(X6[X6 == 5]), len(X6[X6 == 6]),
              len(X6[X6 == 7]), len(X6[X6 == 8]), len(X6[X6 == 9]),
              len(X6[X6 == 10])]
    plt.bar([x + width for x in x_axis], y_axis, width, color='green', label="Benign")
    plt.xlabel('Single_Epithelial_Cell_Size')
    plt.ylabel('Instance Count of Class')
    plt.legend(loc="best")
    plt.title('Instance Count of Class versus Single_Epithelial_Cell_Size')
    plt.savefig('Graph/singe_epithelial_cell_size')
    plt.close()

    X7 = df.processed_data[df.processed_data['Class'] != 2]['Normal_Nucleoli']
    y_axis = [len(X7[X7 == 1]), len(X7[X7 == 2]), len(X7[X7 == 3]),
              len(X7[X7 == 4]), len(X7[X7 == 5]), len(X7[X7 == 6]),
              len(X7[X7 == 7]), len(X7[X7 == 8]), len(X7[X7 == 9]),
              len(X7[X7 == 10])]
    plt.bar(x_axis, y_axis, width, color='orange', label="Malignant")
    X7 = df.processed_data[df.processed_data['Class'] != 4]['Normal_Nucleoli']
    y_axis = [len(X7[X7 == 1]), len(X7[X7 == 2]), len(X7[X7 == 3]),
              len(X7[X7 == 4]), len(X7[X7 == 5]), len(X7[X7 == 6]),
              len(X7[X7 == 7]), len(X7[X7 == 8]), len(X7[X7 == 9]),
              len(X7[X7 == 10])]
    plt.bar([x + width for x in x_axis], y_axis, width, color='green', label="Benign")
    plt.xlabel('Normal_Nucleoli')
    plt.ylabel('Instance Count of Class')
    plt.legend(loc="best")
    plt.title('Instance Count of Class versus Normal_Nucleoli')
    plt.savefig('Graph/normal_nuclei')
    plt.close()

    X8 = df.processed_data[df.processed_data['Class'] != 2]['Bare_Nuclei']
    y_axis = [len(X8[X8 == 1]), len(X8[X8 == 2]), len(X8[X8 == 3]),
              len(X8[X8 == 4]), len(X8[X8 == 5]), len(X8[X8 == 6]),
              len(X8[X8 == 7]), len(X8[X8 == 8]), len(X8[X8 == 9]),
              len(X8[X8 == 10])]
    plt.bar(x_axis, y_axis, width, color='orange', label="Malignant")
    X8 = df.processed_data[df.processed_data['Class'] != 4]['Bare_Nuclei']
    y_axis = [len(X8[X8 == 1]), len(X8[X8 == 2]), len(X8[X8 == 3]),
              len(X8[X8 == 4]), len(X8[X8 == 5]), len(X8[X8 == 6]),
              len(X8[X8 == 7]), len(X8[X8 == 8]), len(X8[X8 == 9]),
              len(X8[X8 == 10])]
    plt.bar([x + width for x in x_axis], y_axis, width, color='green', label="Benign")
    plt.xlabel('Bare_Nuclei')
    plt.ylabel('Instance Count of Class')
    plt.legend(loc="best")
    plt.title('Instance Count of Class versus Bare_Nuclei')
    plt.savefig('Graph/bare_nuclei')
    plt.close()