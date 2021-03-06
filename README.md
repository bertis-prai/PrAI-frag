<h1>PrAI-frag</h1>

Deep learning model for peptide y-ion fragmentation prediction<br>
You can run the pre-trained model on 
<a href="http://www.prai.co.kr/" target="_blank">Website</a>.
<br>

<h2>Enviroment</h2>
<ul>
 <li>python ver. 3.8.3 using python libraries</li>
 <li>torch ver. 1.7.1</li>
 <li>numpy ver. 1.18.5</li>
 <li>pandas ver. 1.2.2</li>
 <li>pyteomics ver.4.3.2</li>
 <li>sklearn ver. 0.24</li>
 <li>pyYAML ver. 5.4.1</li>
 <li>easydict ver. 1.9</li>
</ul>
Run <code>pip install -r requirements.txt</code> in terminal, if you want to install enviroment.
<br>
<br>

<h2>How to Use</h2>
<h4>Training</h4>
To reproduce the results in the manuscript just run <code>training.py</code> on console.<br>
To train with different data, make changes in the trianing file from the config.yaml. Trianing file should be in the same format as the original csv file.  
</code></pre>
Trained models will be saved in <code>logs/</code>.
<br>
<br>

<h4>Inference</h4>
To predict using the PrAI-frag, either use the webiste or use the provided <code>inference.py</code>.
The input file should be in the following format.
<ul>
 <li>File type should be <code>*.csv</code></li>
 <li>
  File must have 3 columns named 
  <code>Peptide</code>,
  <code>Charge</code>,
  <code>CE</code>.
 </li>
 <li><code>CE</code> and <code>Charge</code> value will be automatically calculated, if unsubmitted</li>
</ul>
<table>
 <th>Peptide</th><th>Charge</th><th>CE</th>
 <tr>
  <td>AAAAAAAAAK</td><td>2</td><td>24.6086</td>
 </tr>
 <tr>
  <td>AAAAAAAAAR</td><td>2</td><td></td>
 </tr>
 <tr>
  <td>AAAAAAAVSR</td><td></td><td>31.5383</td>
 </tr>
 <tr>
  <td>...</td><td>...</td><td>...</td>
 </tr>
 <tr>
  <td>AAAACLDK</td><td></td><td></td>
 </tr>
</table>
<br>
To infer a different data fill in <code>config.yaml</code> .
<pre><code> ### INPUT ###

 ...
 
 INFER_DATA: '{your workspace}/input/Testset_data(NIST-rat).csv' # <-- use dafault testset or fill in path of your data</code></pre>
Run <code>inference.py</code> on console.<br>
<pre><code>python worksapce/src/inference.py</code></pre>

If run was successful, <code>{your input file name}_pred.csv</code> will be created at <code>{your input file path}/{your input file name}_pred.csv</code>.
<table>
 <th>Peptide</th><th>Charge</th><th>CE</th><th>y1</th><th>y1^2</th><th>...</th><th>y14^2</th>
 <tr>
  <td>AAAAAAAAAK</td><td>2</td><td>24.6086</td><td></td><td></td><td>...</td><td></td>
 </tr>
 <tr>
  <td>AAAAAAAAAR</td><td>2</td><td>25.9651</td><td></td><td></td><td>...</td><td></td>
 </tr>
 <tr>
  <td>AAAAAAAVSR</td><td>2</td><td>31.5383</td><td></td><td></td><td>...</td><td></td>
 </tr>
 <tr>
  <td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td>
 </tr>
 <tr>
  <td>AAAACLDK</td><td>2</td><td>26.1567</td><td></td><td></td><td>...</td><td></td>
 </tr>
</table>

<br>
<br>
<h4>Calculate PCC</h4>
First, open <code>{workplace}/src/result_PCC.ipynb</code> on with Jupyter notebook.

<br>
Second, fill in the data path.
<ul>
 <li>File type should be <code>*.csv</code></li>
 <li>
  File must have 2 columns named 
  <code>Peptide</code>,
  <code>Charge</code>.
 </li>
 <li>The order of the target data's row and the order of the prediction data's row should be the same.</li>
 <li>The target file of testset(NIST_rat) is located <code>/data/Testset_data(NIST_rat)_target.csv</code>.</li>
</ul>
<pre><code> ''' 
 Read target data & predction data
     - The order of the target data'row and the order of prediction data'row
       should be same.
 '''
# target = pd.read_csv("{your target file's path}")
# pred = pd.read_csv("{your prediction file's path}")</code></pre>
To reproduce the data from manuscript
<pre><code>'''
Prosit & MS2PIP results can be parsed as follows,
'''
### Prosit result
# prosit_result = pd.read_csv("{path of prosit result}")
# parsed_prosit_result = parse_prosit_result(prosit_result)
### MS2PIP
# ms2pip_result = pd.read_csv("{path of ms2pip result}")
# parsed_ms2pip_result = parse_ms2pip_result(ms2pip_result)</code></pre>

<br>
Third, run the calculating cell.
<pre><code> '''
 Calculte PCC and create table
 '''
# get_pcc(target, pred)</code></pre>
<code>get_pcc(target, pred)</code> returns PCC data frame.<br>
<table>
 <th>Peptide</th><th>Charge</th><th>PCC</th>
 <tr>
  <td>AAAAAAAAAK</td><td>2</td><td></td>
 </tr>
 <tr>
  <td>AAAAAAAAAR</td><td>2</td><td></td>
 </tr>
 <tr>
  <td>AAAAAAAVSR</td><td>2</td><td></td>
 </tr>
 <tr>
  <td>...</td><td>...</td><td>...</td>
 </tr>
 <tr>
  <td>AAAACLDK</td><td>2</td><td></td>
 </tr>
</table>

