import {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	NodeOperationError,
} from 'n8n-workflow';

export class AiAssistantMemory implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'AI Assistant Memory',
		name: 'aiAssistantMemory',
		icon: 'file:aiassistant.svg',
		group: ['ai'],
		version: 1,
		subtitle: '={{$parameter["operation"]}}',
		description: 'Manage AI Assistant memory and knowledge storage',
		defaults: {
			name: 'AI Assistant Memory',
		},
		inputs: ['main'],
		outputs: ['main'],
		credentials: [
			{
				name: 'aiAssistantApi',
				required: true,
			},
		],
		requestDefaults: {
			baseURL: '={{$credentials.baseUrl}}/api/v1',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
			},
		},
		properties: [
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				noDataExpression: true,
				options: [
					{
						name: 'Store Fact',
						value: 'storeFact',
						description: 'Store an important fact in memory',
						action: 'Store a fact in memory',
					},
					{
						name: 'Retrieve Memory',
						value: 'retrieveMemory',
						description: 'Retrieve relevant memories based on query',
						action: 'Retrieve memory',
					},
					{
						name: 'Search Memory',
						value: 'searchMemory',
						description: 'Search through stored memories',
						action: 'Search memory',
					},
					{
						name: 'Update Memory',
						value: 'updateMemory',
						description: 'Update an existing memory entry',
						action: 'Update memory',
					},
					{
						name: 'Delete Memory',
						value: 'deleteMemory',
						description: 'Delete a memory entry',
						action: 'Delete memory',
					},
				],
				default: 'storeFact',
			},
			// Store Fact Parameters
			{
				displayName: 'Fact',
				name: 'fact',
				type: 'string',
				required: true,
				displayOptions: {
					show: {
						operation: ['storeFact'],
					},
				},
				default: '',
				placeholder: 'Enter the fact to store',
				description: 'The fact or information to store in memory',
			},
			{
				displayName: 'Session ID',
				name: 'sessionId',
				type: 'string',
				required: true,
				displayOptions: {
					show: {
						operation: ['storeFact'],
					},
				},
				default: '',
				description: 'Session ID to associate with this memory',
			},
			{
				displayName: 'Importance',
				name: 'importance',
				type: 'number',
				displayOptions: {
					show: {
						operation: ['storeFact'],
					},
				},
				default: 0.7,
				typeOptions: {
					minValue: 0,
					maxValue: 1,
					numberStepSize: 0.1,
				},
				description: 'Importance score (0-1) for this memory',
			},
			{
				displayName: 'Tags',
				name: 'tags',
				type: 'string',
				displayOptions: {
					show: {
						operation: ['storeFact'],
					},
				},
				default: '',
				placeholder: 'tag1, tag2, tag3',
				description: 'Comma-separated tags for categorizing this memory',
			},
			// Retrieve Memory Parameters
			{
				displayName: 'Query',
				name: 'query',
				type: 'string',
				required: true,
				displayOptions: {
					show: {
						operation: ['retrieveMemory', 'searchMemory'],
					},
				},
				default: '',
				placeholder: 'Enter your search query',
				description: 'Query to search for relevant memories',
			},
			{
				displayName: 'Session ID',
				name: 'sessionId',
				type: 'string',
				displayOptions: {
					show: {
						operation: ['retrieveMemory', 'searchMemory'],
					},
				},
				default: '',
				description: 'Session ID to scope the search (optional)',
			},
			{
				displayName: 'Max Results',
				name: 'maxResults',
				type: 'number',
				displayOptions: {
					show: {
						operation: ['retrieveMemory', 'searchMemory'],
					},
				},
				default: 10,
				typeOptions: {
					minValue: 1,
					maxValue: 100,
				},
				description: 'Maximum number of results to return',
			},
			{
				displayName: 'Similarity Threshold',
				name: 'similarityThreshold',
				type: 'number',
				displayOptions: {
					show: {
						operation: ['retrieveMemory', 'searchMemory'],
					},
				},
				default: 0.5,
				typeOptions: {
					minValue: 0,
					maxValue: 1,
					numberStepSize: 0.1,
				},
				description: 'Minimum similarity score for results (0-1)',
			},
			// Update Memory Parameters
			{
				displayName: 'Memory ID',
				name: 'memoryId',
				type: 'string',
				required: true,
				displayOptions: {
					show: {
						operation: ['updateMemory', 'deleteMemory'],
					},
				},
				default: '',
				description: 'ID of the memory entry to update or delete',
			},
			{
				displayName: 'Updated Content',
				name: 'updatedContent',
				type: 'string',
				displayOptions: {
					show: {
						operation: ['updateMemory'],
					},
				},
				default: '',
				description: 'Updated content for the memory entry',
			},
			{
				displayName: 'Updated Importance',
				name: 'updatedImportance',
				type: 'number',
				displayOptions: {
					show: {
						operation: ['updateMemory'],
					},
				},
				default: 0.7,
				typeOptions: {
					minValue: 0,
					maxValue: 1,
					numberStepSize: 0.1,
				},
				description: 'Updated importance score (0-1)',
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];

		for (let i = 0; i < items.length; i++) {
			const operation = this.getNodeParameter('operation', i) as string;

			let responseData;

			try {
				if (operation === 'storeFact') {
					const fact = this.getNodeParameter('fact', i) as string;
					const sessionId = this.getNodeParameter('sessionId', i) as string;
					const importance = this.getNodeParameter('importance', i) as number;
					const tagsString = this.getNodeParameter('tags', i) as string;

					const tags = tagsString
						? tagsString.split(',').map((tag) => tag.trim()).filter((tag) => tag.length > 0)
						: [];

					const body = {
						fact,
						session_id: sessionId,
						importance,
						tags,
					};

					responseData = await this.helpers.requestWithAuthentication.call(
						this,
						'aiAssistantApi',
						{
							method: 'POST',
							url: '/memory/store',
							body,
							json: true,
						},
					);
				} else if (operation === 'retrieveMemory') {
					const query = this.getNodeParameter('query', i) as string;
					const sessionId = this.getNodeParameter('sessionId', i) as string;
					const maxResults = this.getNodeParameter('maxResults', i) as number;
					const similarityThreshold = this.getNodeParameter('similarityThreshold', i) as number;

					const params = new URLSearchParams({
						query,
						max_results: maxResults.toString(),
						similarity_threshold: similarityThreshold.toString(),
					});

					if (sessionId) {
						params.append('session_id', sessionId);
					}

					responseData = await this.helpers.requestWithAuthentication.call(
						this,
						'aiAssistantApi',
						{
							method: 'GET',
							url: `/memory/retrieve?${params.toString()}`,
							json: true,
						},
					);
				} else if (operation === 'searchMemory') {
					const query = this.getNodeParameter('query', i) as string;
					const sessionId = this.getNodeParameter('sessionId', i) as string;
					const maxResults = this.getNodeParameter('maxResults', i) as number;
					const similarityThreshold = this.getNodeParameter('similarityThreshold', i) as number;

					const params = new URLSearchParams({
						q: query,
						limit: maxResults.toString(),
						threshold: similarityThreshold.toString(),
					});

					if (sessionId) {
						params.append('session_id', sessionId);
					}

					responseData = await this.helpers.requestWithAuthentication.call(
						this,
						'aiAssistantApi',
						{
							method: 'GET',
							url: `/memory/search?${params.toString()}`,
							json: true,
						},
					);
				} else if (operation === 'updateMemory') {
					const memoryId = this.getNodeParameter('memoryId', i) as string;
					const updatedContent = this.getNodeParameter('updatedContent', i) as string;
					const updatedImportance = this.getNodeParameter('updatedImportance', i) as number;

					const body = {
						content: updatedContent,
						importance: updatedImportance,
					};

					responseData = await this.helpers.requestWithAuthentication.call(
						this,
						'aiAssistantApi',
						{
							method: 'PUT',
							url: `/memory/${memoryId}`,
							body,
							json: true,
						},
					);
				} else if (operation === 'deleteMemory') {
					const memoryId = this.getNodeParameter('memoryId', i) as string;

					responseData = await this.helpers.requestWithAuthentication.call(
						this,
						'aiAssistantApi',
						{
							method: 'DELETE',
							url: `/memory/${memoryId}`,
							json: true,
						},
					);
				}

				const executionData = this.helpers.constructExecutionMetaData(
					this.helpers.returnJsonArray(responseData as INodeExecutionData),
					{ itemData: { item: i } },
				);

				returnData.push(...executionData);
			} catch (error) {
				if (this.continueOnFail()) {
					const executionData = this.helpers.constructExecutionMetaData(
						this.helpers.returnJsonArray({ error: error.message }),
						{ itemData: { item: i } },
					);
					returnData.push(...executionData);
					continue;
				}
				throw error;
			}
		}

		return [returnData];
	}
}