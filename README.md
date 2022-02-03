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
<br>

<h2>How to Use</h2>
<h4>Training</h4>
if you want to train model, you just run <code>training.py</code> on console.<br>
<pre><code>
python worksapce/src/training.py

</code></pre>
Trained models will be saved in <code>logs/</code>.
<br>
<br>

<h4>Inference</h4>
First, check your data format.
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
Second, you should fill in <code>config.yaml</code>, if you want to infer your data.
<pre><code>
 ### INPUT ###

 ...
 
 INFER_DATA: '{your workspace}/input/Testset_data(NIST-rat).csv' # <-- use dafault testset or fill in path of your data
</code></pre>
<br>
Third, run <code>inference.py</code> on console.<br>
<pre><code>
python worksapce/src/inference.py

</code></pre>
And Then <code>{your input file name}_pred.csv</code> will be created at <code>{your input file path}/{your input file name}_pred.csv</code>.
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
First, open <code>{workplace}/src/result_PCC.ipynb</code> on your jupyter notebook.

<br>
Second, fill in path of data.
<ul>
 <li>File type should be <code>*.csv</code></li>
 <li>
  File must have 2 columns named 
  <code>Peptide</code>,
  <code>Charge</code>.
 </li>
 <li>The order of the target data's row and the order of the prediction data's row should be the same.</li>
 <li>target file of testset(NIST_rat) is located <code>/data/Testset_data(NIST_rat)_target.csv</code></li>
</ul>
<pre><code>
 ''' 
 Read target data & predction data
     - The order of the target data'row and the order of prediction data'row
       should be same.
 '''
# target = pd.read_csv("{your target file's path}")
# pred = pd.read_csv("{your prediction file's path}")

</code></pre>

<br>
Third, run the calculating cell.
<pre><code>
 '''
 Calculte PCC and create table
 '''
# get_pcc(target, pred)

</code></pre>
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
  <td>AAAAAAAVSR</td><td></td><td></td>
 </tr>
 <tr>
  <td>...</td><td>...</td><td>...</td>
 </tr>
 <tr>
  <td>AAAACLDK</td><td></td><td></td>
 </tr>
</table>

