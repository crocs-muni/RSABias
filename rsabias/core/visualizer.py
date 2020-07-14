import subprocess
import pandas as pd
import os
import json
import statistics


class ClassificationTableVisualizer:
    def __init__(self, results_path, output_path, template_path):
        self.results_filepath = results_path
        self.output_filepath = output_path
        self.template_filepath = template_path
        self.data_filepaths = []
        self.n_cols = 0
        self.n_keys_params = []
        self.top_n_params = []
        self.accuracy_data = {}
        self.groups_order = []
        self.groups_mapping = {}

    @staticmethod
    def get_tabular_layour_string(n_cols):
        """
        Based on how many columns the table should have, prepares a line with layout for tabular tag in latex with an
        extra column.
        :param n_cols: how many columns the data has = how many n_keys parameter configurations exist = how many files in the folder with json data
        """
        token = '@{}c@{}|'
        layout = '|'
        for _ in range(n_cols + 1):
            layout += token
        return layout

    @staticmethod
    def get_columns_label_line_string(n_cols):
        """
        Prepares the header label of table, generates something like:
        & Top 1 match & Top 2 match & top 3 match \\ - texline for n_cols = 3
        :param n_cols:
        :return:
        """
        layout = ''
        for i in range(n_cols):
            label = '& Top ' + str(i + 1) + ' match '
            layout += label
        layout += '\\\\'
        return layout

    def get_tabular_definition_line_string(self, n_cols):
        """
        Prepare whole line with table definition into the latex file.
        Example: \begin{tabular}{|@{}c@{}|@{}c@{}|} \n
        """
        tabular_line = '\t\\begin{tabular}{' + self.get_tabular_layour_string(n_cols) + '}\n'
        h_line = '\t\t\\hline\n'
        first_row_line = '\t\t\t' + self.get_columns_label_line_string(n_cols)
        return tabular_line + h_line + first_row_line

    @staticmethod
    def compile_latex(filepath_to_latex, workdir):
        """
        Runs pdflatex on the table.tex file
        :param filepath_to_latex: on what file to run the latex.
        :param workdir: what is the workdir (same where the latex file is sitting)
        :return: Nothing
        """
        proc = subprocess.Popen(['pdflatex', filepath_to_latex], cwd=workdir)
        proc.communicate()

    @staticmethod
    def get_all_json_files_in_dir(dirpath):
        """
        Returns filepath to all json files in the dirpath parameter
        """
        files = []
        for file in os.listdir(dirpath):
            if file.endswith('.json'):
                files.append(os.path.join(dirpath, file))
        return files

    @staticmethod
    def get_n_keys_params(filepaths):
        """
        Parses all configurations of "how many keys in one batch" we have the data for
        """
        params = []
        for f in filepaths:
            with open(f) as json_file:
                data = json.load(json_file)
            params.append(int(data['n_keys']))
        return sorted(params)

    @staticmethod
    def get_top_n_params(filepath):
        """
        Parses all configurations of "How many top results we check" we have the data for
        """
        params = []
        with open(filepath) as json_file:
            data = json.load(json_file)
        for top_n in data['accuracies'].keys():
            params.append(int(top_n))
        return params

    def get_accuracies_from_json(self, input_filepath):
        """
        Fills the accuracies dictionary from the json files
        """
        with open(input_filepath) as json_file:
            data = json.load(json_file)
            key_name = 'n_keys_' + str(data['n_keys'])
            for top_n_match, top_n_acc in data['accuracies'].items():
                dct_to_fill = 'top_n_' + str(top_n_match)
                accuracies = []
                for group_name, group_accuracy in top_n_acc.items():
                    total = group_accuracy['correct'] + group_accuracy['wrong']
                    acc = group_accuracy['correct'] / total if total > 0 else float(0)
                    acc *= 100
                    accuracies.append(round(acc, 1))
                accuracies.append(round(statistics.mean(accuracies), 1))
                self.accuracy_data[dct_to_fill][key_name] = accuracies

    def generate_groups_csv_from_json(self):
        """
        Generates list of groups from json file.
        """
        n_groups = 0
        output_filepath = os.path.join(self.output_filepath, 'groups.csv')
        with open(self.data_filepaths[0]) as json_file:
            data = json.load(json_file)
            col_params = []
            for top_n_match, top_n_acc in data['accuracies'].items():
                col_params.append(int(top_n_match))
                n_groups = len(top_n_acc)

                for group_name, group_accuracy in top_n_acc.items():
                    self.groups_order.append(int(group_name))
                break

        self.groups_order.append(999)  # TODO: Very ugly hack. Making sure that Average numbers appear on the last line

        groups = ['{Group ' + str(i) + '}' for i in range(1, n_groups + 1)]
        groups.append('Average')
        dct = {'batch_keys': groups}
        df = pd.DataFrame(data=dct)
        df.to_csv(output_filepath, sep=',', index=False, header=True)

    def get_accuracy_data_from_jsons(self):
        for f in self.data_filepaths:
            self.get_accuracies_from_json(f)

    def dump_csvs(self):
        self.generate_groups_csv_from_json()

        for top_n_param, dct in self.accuracy_data.items():
            df = pd.DataFrame(data=dct)

            # rearrange columns
            cols = df.columns.to_list()
            nums = [int(x.split('_')[2]) for x in cols]
            nums, reordered = (list(x) for x in zip(*sorted(zip(nums, cols), key=lambda pair: pair[0])))
            df = df[reordered]
            df['group'] = self.groups_order
            df = df.sort_values(by=['group'])
            df = df.drop(columns=['group'])
            # dump
            df.to_csv(os.path.join(self.output_filepath, top_n_param + '.csv'), header=True, index=False, sep=',')

    def delete_auxilary_files(self):
        files = os.listdir(self.output_filepath)

        for f in files:
            if f.endswith('.aux'):
                os.remove(os.path.join(self.output_filepath, f))
            if f.endswith('.log'):
                os.remove(os.path.join(self.output_filepath, f))

    def extract_table_parameters_from_json_files(self):
        return self.get_n_keys_params(self.data_filepaths), self.get_top_n_params(self.data_filepaths[0])

    def parse_json_data(self):
        self.data_filepaths = self.get_all_json_files_in_dir(self.results_filepath)
        self.n_keys_params, self.top_n_params = self.extract_table_parameters_from_json_files()
        self.n_cols = len(self.top_n_params)
        self.accuracy_data = {'top_n_' + str(x): {} for x in self.top_n_params}
        self.get_accuracy_data_from_jsons()
        self.dump_csvs()

    @staticmethod
    def get_table_column_name_line(param):
        token_1 = '\t\t\t\tcolumns/n_keys_'
        token_3 = '/.style = {column name={'
        token_5 = '}, string type,},\n'
        return token_1 + str(param) + token_3 + str(param) + token_5

    def get_table_template(self, top_n):
        with open(os.path.join(self.template_filepath, 'data_table_template_prefix.txt')) as table_template_prefix:
            table_template_prefix = table_template_prefix.read()
        with open(os.path.join(self.template_filepath, 'data_table_template_suffix.txt')) as table_template_suffix:
            table_template_suffix = table_template_suffix.read()

        data = '\t\t\t& ' + table_template_prefix

        for n in self.n_keys_params:
            data += self.get_table_column_name_line(n)
        data += table_template_suffix + '{top_n_' + str(top_n) + '.csv}}\n'
        return data

    def construct_table(self):
        """
        Whole TeX file is one string:
            top_data \n
            get_tabular_lines(n_cons) \n
            groups_table_template + {path_to_groups_csv} \n
            n_cols times: & \n data_table_template + {path_to_data_csv} \n
            bottom_data
        """
        self.parse_json_data()

        with open(os.path.join(self.template_filepath, 'top.txt')) as top:
            top_data = top.read()

        with open(os.path.join(self.template_filepath, 'bottom.txt')) as bottom:
            bottom_data = bottom.read()

        with open(os.path.join(self.template_filepath, 'groups_table_template.txt')) as groups_table_template:
            groups_table_template = groups_table_template.read()

        tex_source = ''
        tex_source += top_data + '\n'
        tex_source += self.get_tabular_definition_line_string(self.n_cols) + '\n'
        tex_source += '\t\t\t' + groups_table_template + '{groups.csv}}\n'

        for i in self.top_n_params:
            tex_source += self.get_table_template(i)
        tex_source += bottom_data

        with open(os.path.join(self.output_filepath, 'table.tex'), 'w') as tex_file:
            tex_file.write(tex_source)

        self.compile_latex(os.path.join(self.output_filepath, 'table.tex'), self.output_filepath)
        self.delete_auxilary_files()


