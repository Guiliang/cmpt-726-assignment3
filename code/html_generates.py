def generate_html(score_list):
    """
    generate html table
    :param score_list: the probability of classifying
    :return: html table
    """
    html = "<html>\n<body>\n\n<table border=>\n\t" \
           "<tr>\n" \
           "\t\t<th>Month</th>\n" \
           "\t\t<th>Savings</th>\n" \
           "\t</tr>\n" \
           "\t<tr>\n" \
           "\t\t<td>January</td>\n" \
           "\t\t<td>$100</td>\n" \
           "\t</tr>\n" \
           "</table>\n\n" \
           "</body>\n" \
           "</html>"

    html_line = []
    html_line.append("<html>\n")
    html_line.append("<body>\n\n")
    html_line.append("<h1>Test Image and Its Classification Scores</h1>\n")
    html_line.append("<table border=>\n\t")

    html_line.append("<tr>\n")
    html_line.append("\t\t<th>Image</th>\n")
    html_line.append("\t\t<th>P(Basketball)</th>\n")
    html_line.append("\t\t<th>P(Hockey)</th>\n")
    html_line.append("\t\t<th>P(Soccer)</th>\n")
    html_line.append("\t\t<th>The prediction</th>\n")
    html_line.append("\t</tr>\n")

    predict_target = ["Basketball", "Hockey", "Soccer"]

    for score in score_list:
        score_number = score[:3]
        max_place = score_number.index(max(score_number))
        predict_result = predict_target[max_place]
        score_number
        html_line.append("<tr>\n")
        html_line.append(
            '\t\t<th><img src ="' + str(score[3]) + '"></th>\n')
        html_line.append("\t\t<th>" + str(score[0]) + "</th>\n")
        html_line.append("\t\t<th>" + str(score[1]) + "</th>\n")
        html_line.append("\t\t<th>" + str(score[2]) + "</th>\n")
        html_line.append("\t\t<th>" + str(predict_result) + "</th>\n")
        html_line.append("\t</tr>\n")

    html_line.append("</table>\n\n")
    html_line.append("</body>\n")
    html_line.append("</html>")

    return html_line


def write_lines(line_data_list, file_name):
    """
    write data to file
    :param line_data_list: data line list
    :param file_name: file_name
    :return: nothing
    """
    file_object = open(file_name, 'w')
    for line in line_data_list:
        file_object.write(line)
    file_object.close()

# test
# html = generate_html([["/Users/liu/Desktop/CMPT 726 Machine Learning material/sport3/validation/hockey/img_2997.jpg",0.5, 0.3, 0.2]])
# write_lines(html, "predict_table.html")
