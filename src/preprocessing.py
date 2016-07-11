
# coding: utf-8

# ## Describing Variables

# In[74]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
get_ipython().magic(u'matplotlib inline')
pd.options.display.max_columns = 250


# In[36]:

# import all 21 csvs

bill_of_materials_df = pd.read_csv('../competition_data/bill_of_materials.csv')
comp_adaptor_df = pd.read_csv('../competition_data/comp_adaptor.csv')
comp_boss_df = pd.read_csv('../competition_data/comp_boss.csv')
comp_elbow_df = pd.read_csv('../competition_data/comp_elbow.csv')
comp_float_df = pd.read_csv('../competition_data/comp_float.csv')
comp_hfl_df = pd.read_csv('../competition_data/comp_hfl.csv')
comp_nut_df = pd.read_csv('../competition_data/comp_nut.csv')
comp_other_df = pd.read_csv('../competition_data/comp_other.csv')
comp_sleeve_df = pd.read_csv('../competition_data/comp_sleeve.csv')
comp_straight_df = pd.read_csv('../competition_data/comp_straight.csv')
comp_tee_df = pd.read_csv('../competition_data/comp_tee.csv')
comp_threaded_df = pd.read_csv('../competition_data/comp_threaded.csv')
components_df = pd.read_csv('../competition_data/components.csv')
specs_df = pd.read_csv('../competition_data/specs.csv')
test_set_df = pd.read_csv('../competition_data/test_set.csv')
train_set_df = pd.read_csv('../competition_data/train_set.csv')
tube_end_form_df = pd.read_csv('../competition_data/tube_end_form.csv')
tube_df = pd.read_csv('../competition_data/tube.csv')
type_component_df = pd.read_csv('../competition_data/type_component.csv')
type_connection_df = pd.read_csv('../competition_data/type_connection.csv')
type_end_form_df = pd.read_csv('../competition_data/type_end_form.csv')


# In[37]:

# prepare test_set_df and train_set_df for concatenation
test_id_series = test_set_df['id']
test_set_df.drop(labels = 'id', axis = 1, inplace = True)
train_cost_series = train_set_df['cost']
train_set_df.drop(labels = 'cost', axis = 1, inplace = True)

#concatenate test_set_df and train_set_df to create master_df, with test_set_df on top
master_df = pd.concat(objs = [test_set_df, train_set_df], axis = 0)

# next, format appropriately master_df
master_df['tube_assembly_id'] = pd.Series(master_df['tube_assembly_id'], dtype = 'category')
master_df['supplier'] = pd.Series(master_df['supplier'], dtype = 'category')
master_df['bracket_pricing'] = pd.Series(master_df['bracket_pricing'], dtype = 'category')
df = pd.to_datetime(master_df['quote_date'])
df = df.to_frame()
df['first_date'] = pd.Timestamp('19820922')
master_df['quote_date'] = (df['quote_date'] - df['first_date']).dt.days


# In[38]:

# format tube_df to prepare for joining with master_df
tube_df['tube_assembly_id'] = pd.Series(tube_df['tube_assembly_id'], dtype = 'category')
tube_df['material_id'] = pd.Series(tube_df['material_id'], dtype = 'category')
tube_df['end_a_1x'] = pd.Series(tube_df['end_a_1x'], dtype = 'category')
tube_df['end_a_2x'] = pd.Series(tube_df['end_a_2x'], dtype = 'category')
tube_df['end_x_1x'] = pd.Series(tube_df['end_x_1x'], dtype = 'category')
tube_df['end_x_2x'] = pd.Series(tube_df['end_x_2x'], dtype = 'category')
tube_df['end_a'] = pd.Series(tube_df['end_a'], dtype = 'category')
tube_df['end_x'] = pd.Series(tube_df['end_x'], dtype = 'category')
# replace 9999.0 entries in bend_radius column with np.nan entries
tube_df = tube_df.replace(9999.0, np.nan)
tube_df = tube_df.replace('9999', 'other')


# In[39]:

# left join master_df with tube_df
result_1 = pd.merge(master_df,
                    tube_df,
                    left_on = 'tube_assembly_id',
                    right_on = 'tube_assembly_id',
                    how='left',
                    sort=False)

# left join with left = result_1 (master and tube) and right = specs_with_totals
specs_only_df = specs_df.iloc[:, 1:11]
specs_logical_df = ~specs_only_df.isnull()
spec_totals = specs_logical_df.sum(axis = 1)
specs_with_totals_df = specs_df.copy()
specs_with_totals_df['spec_totals'] = spec_totals
result_2 = result_1.merge(specs_with_totals_df[['tube_assembly_id', 'spec_totals']])

# left join with left = result_2 (train, tube, and specs_with_totals) and right = bill_of_materials_summary_df
bill_comp_types_df = bill_of_materials_df.iloc[:, (1,3,5,7,9,11,13,15)]
bill_comp_types_logical_df = ~bill_comp_types_df.isnull()
component_series = bill_comp_types_logical_df.sum(axis = 1)
bill_comp_quants_df = bill_of_materials_df.iloc[:, (2,4,6,8,10,12,14,16)]
quants_series = bill_comp_quants_df.sum(axis = 1)
bill_of_materials_summary_df = bill_of_materials_df.copy()
bill_of_materials_summary_df['type_totals'] = component_series
bill_of_materials_summary_df['component_totals'] = quants_series
result_3 = result_2.merge(bill_of_materials_summary_df[['tube_assembly_id', 'type_totals', 'component_totals']])
master_df = result_3.copy()


# In[40]:

# replace supplier column with indicator variable columns
supplier_series = master_df['supplier']
supplier_dummy_df = pd.get_dummies(data = supplier_series, prefix = 'supp', prefix_sep = '_')
master_df.drop(labels = 'supplier', axis = 1, inplace = True)
master_df = pd.concat(objs = [master_df, supplier_dummy_df], axis = 1)


# In[41]:

# convert N/Y bracket pricing column to 0/1 
bracket_pricing_series = master_df['bracket_pricing']
master_df['bracket_pricing'] = bracket_pricing_series.replace(['No', 'Yes'], [0, 1])


# In[42]:

# replace material_id column with dummy variable columns, including NA
material_id_series = master_df['material_id']
material_id_dummy_df = pd.get_dummies(data = material_id_series, prefix = 'mat', prefix_sep = '_', dummy_na = True)
master_df.drop(labels = 'material_id', axis = 1, inplace = True)
master_df = pd.concat(objs = [master_df, material_id_dummy_df], axis = 1)


# In[43]:

# convert end_a_1x, end_a_2x, end_x_1x, end_x_2x columns to 0/1. 
master_df['end_a_1x'] = master_df['end_a_1x'].replace(['N', 'Y'], [0, 1])
master_df['end_a_2x'] = master_df['end_a_2x'].replace(['N', 'Y'], [0, 1])
master_df['end_x_1x'] = master_df['end_x_1x'].replace(['N', 'Y'], [0, 1])
master_df['end_x_2x'] = master_df['end_x_2x'].replace(['N', 'Y'], [0, 1])


# In[44]:

# replace end_a column with indicator variable columns
end_a_series = master_df['end_a']
end_a_dummy_df = pd.get_dummies(data = end_a_series, prefix = 'end_a', prefix_sep = '_')
master_df.drop(labels = 'end_a', axis = 1, inplace = True)
master_df = pd.concat(objs = [master_df, end_a_dummy_df], axis = 1)

# replace end_x column with 24 indicator variable columns
end_x_series = master_df['end_x']
end_x_dummy_df = pd.get_dummies(data = end_x_series, prefix = 'end_x', prefix_sep = '_')
master_df.drop(labels = 'end_x', axis = 1, inplace = True)
master_df = pd.concat(objs = [master_df, end_x_dummy_df], axis = 1)


# In[45]:

# it appears that there are 30 NaNs in bend_radius (18 in the test section and 12 in the train section)
# we will impute these 30 values using the mean of the existing values.
bend_radius_mean = master_df['bend_radius'].mean()
master_df = master_df.fillna(value = bend_radius_mean)


# In[46]:

# it appears that there are 2 '9999' component_ids in the bill_of_materials_df. We will
# impute these with the most common component_id in bill_of_materials_df: 'C-1621'.
bill_of_materials_df = bill_of_materials_df.replace('9999', 'C-1621')


# In[47]:

#check whether sets are disjoint
adaptor_set = comp_adaptor_df['component_id'].unique()
boss_set = comp_boss_df['component_id'].unique()
elbow_set = comp_elbow_df['component_id'].unique()
float_set = comp_float_df['component_id'].unique()
hfl_set = comp_hfl_df['component_id'].unique()
nut_set = comp_nut_df['component_id'].unique()
other_set = comp_other_df['component_id'].unique()
sleeve_set = comp_sleeve_df['component_id'].unique()
straight_set = comp_straight_df['component_id'].unique()
tee_set = comp_tee_df['component_id'].unique()
threaded_set = comp_threaded_df['component_id'].unique()

# number of unique component IDs in all data frames
all_components_list = list(adaptor_set) + list(boss_set) + list(elbow_set) + list(float_set) + list(hfl_set) + list(nut_set) + list(other_set) + list(sleeve_set) + list(straight_set) + list(tee_set) + list(threaded_set)
unique_components_list = set(all_components_list)
print 'number of component IDs with repetition: %d' % len(all_components_list)
print 'number of unique component IDs: %d' % len(unique_components_list)


# In[48]:

# build the lookup_component_df
adaptor_str = np.array(['adaptor']* len(adaptor_set))
boss_str = np.array(['boss']* len(boss_set))
elbow_str = np.array(['elbow']* len(elbow_set))
float_str = np.array(['float']* len(float_set))
hfl_str = np.array(['hfl']* len(hfl_set))
nut_str = np.array(['nut']* len(nut_set))
other_str = np.array(['other']* len(other_set))
sleeve_str = np.array(['sleeve']* len(sleeve_set))
straight_str = np.array(['straight']* len(straight_set))
tee_str = np.array(['tee']* len(tee_set))
threaded_str = np.array(['threaded']* len(threaded_set))


# In[49]:

component_id_array = np.concatenate((adaptor_set, boss_set, elbow_set, float_set, hfl_set, nut_set, other_set,                                     sleeve_set, straight_set, tee_set, threaded_set), axis=0)
component_type_array = np.concatenate((adaptor_str, boss_str, elbow_str, float_str, hfl_str, nut_str, other_str,                                     sleeve_str, straight_str, tee_str, threaded_str), axis=0)


# In[50]:

# build empty columns for component counts
component_empty_df = pd.DataFrame(data = np.zeros(shape = (np.shape(bill_of_materials_df)[0], 11), dtype = int),
                                  columns = ['adaptor', 'boss', 'elbow', 'float', 'hfl', 'nut', 'other', 'sleeve',\
                                             'straight', 'tee', 'threaded'])
bill_of_materials_components_df = pd.concat(objs = [bill_of_materials_df, component_empty_df], axis = 1)


# In[51]:

# now fill component counts
component_id_column_names = ['component_id_1', 'component_id_2', 'component_id_3', 'component_id_4',                             'component_id_5', 'component_id_6', 'component_id_7', 'component_id_8']
quantity_column_names = ['quantity_1', 'quantity_2', 'quantity_3', 'quantity_4',                             'quantity_5', 'quantity_6', 'quantity_7', 'quantity_8']
t0 = time()
for tube_assembly_id in bill_of_materials_components_df['tube_assembly_id']:
    row_index = bill_of_materials_components_df[bill_of_materials_components_df['tube_assembly_id']                                                 == tube_assembly_id].index
    row_index = row_index[0]
    for i in range(8):
        component_id = bill_of_materials_components_df[component_id_column_names[i]][row_index]
        quantity = bill_of_materials_components_df[quantity_column_names[i]][row_index]
        if isinstance(component_id, basestring):
            # get component_id type
            component_id_index = np.where(component_id_array == component_id)[0][0]
            component_type = component_type_array[component_id_index]
            bill_of_materials_components_df[component_type][row_index] += quantity
        else:
            break
t1 = time()
print t1-t0


# In[83]:

# to be used instead of running the filling component counts cell as a shortcut
#result_4 = pd.read_csv('./result_4.csv')
#result_4.drop(labels = 'Unnamed: 0', axis = 1, inplace = True)


# In[52]:

# left join master_df with component columns of bill_of_materials_components_df
component_columns_df = bill_of_materials_components_df[['tube_assembly_id',
                                                        'adaptor',
                                                        'boss',
                                                        'elbow',
                                                        'float',
                                                        'hfl',
                                                        'nut',
                                                        'other',
                                                        'sleeve',
                                                        'straight',
                                                        'tee',
                                                        'threaded']]
result_4 = pd.merge(master_df,
                    component_columns_df,
                    left_on = 'tube_assembly_id',
                    right_on = 'tube_assembly_id',
                    how='left',
                    sort=False)


# In[86]:

result_4.drop(labels = 'tube_assembly_id', axis = 1, inplace = True)


# In[89]:

result_4.to_csv('result_4.csv')


# In[91]:

# get column names
column_names = list(result_4.columns)
# write six csvs:
# 1. test_basic.csv (no component counts)
test_basic_df = result_4.iloc[:30235, :158]
test_basic_df.to_csv('test_basic.csv')
# 2. train_basic.csv (no component counts)
train_basic_df = result_4.iloc[30235:, :158]
train_basic_df.to_csv('train_basic.csv')
# 3. test_components.csv (with component counts)
test_components_df = result_4.iloc[:30235, :]
test_components_df.to_csv('test_components.csv')
# 4. train_components.csv (with component counts)
train_components_df = result_4.iloc[30235:, :]
train_components_df.to_csv('train_components.csv')
# 5. test_id.csv
test_id_df = test_id_series.to_frame(name = 'id')
test_id_df.to_csv('test_id.csv')
# 6. responses.csv
cost_df = train_cost_series.to_frame(name = 'cost')
cost_df.to_csv('cost.csv')

