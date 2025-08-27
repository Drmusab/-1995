import {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	NodeOperationError,
} from 'n8n-workflow';

export class AiAssistantWorkflow implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'AI Assistant Workflow',
		name: 'aiAssistantWorkflow',
		icon: 'file:aiassistant.svg',
		group: ['ai'],
		version: 1,
		subtitle: '={{$parameter["operation"]}}',
		description: 'Manage and execute AI Assistant workflows',
		defaults: {
			name: 'AI Assistant Workflow',
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
						name: 'Process Message',
						value: 'processMessage',
						description: 'Process a message through the AI Assistant engine',
						action: 'Process a message',
					},
					{
						name: 'Get Health Status',
						value: 'getHealth',
						description: 'Get system health status',
						action: 'Get health status',
					},
					{
						name: 'Get System Status',
						value: 'getSystemStatus',
						description: 'Get detailed system status information',
						action: 'Get system status',
					},
					{
						name: 'Execute Complex Task',
						value: 'executeTask',
						description: 'Execute a complex multi-step task',
						action: 'Execute a task',
					},
				],
				default: 'processMessage',
			},
			// Process Message Parameters
			{
				displayName: 'Message',
				name: 'message',
				type: 'string',
				required: true,
				displayOptions: {
					show: {
						operation: ['processMessage'],
					},
				},
				default: '',
				placeholder: 'Enter your message',
				description: 'Message to process through the AI Assistant',
			},
			{
				displayName: 'Session ID',
				name: 'sessionId',
				type: 'string',
				displayOptions: {
					show: {
						operation: ['processMessage', 'executeTask'],
					},
				},
				default: '',
				description: 'Session ID for context (optional)',
			},
			{
				displayName: 'User ID',
				name: 'userId',
				type: 'string',
				displayOptions: {
					show: {
						operation: ['processMessage', 'executeTask'],
					},
				},
				default: '',
				description: 'User ID for personalization (optional)',
			},
			{
				displayName: 'Context Data',
				name: 'contextData',
				type: 'json',
				displayOptions: {
					show: {
						operation: ['processMessage', 'executeTask'],
					},
				},
				default: '{}',
				description: 'Additional context data as JSON',
			},
			{
				displayName: 'Processing Options',
				name: 'processingOptions',
				type: 'collection',
				placeholder: 'Add Option',
				displayOptions: {
					show: {
						operation: ['processMessage'],
					},
				},
				default: {},
				options: [
					{
						displayName: 'Enable Memory',
						name: 'enableMemory',
						type: 'boolean',
						default: true,
						description: 'Whether to use memory for this request',
					},
					{
						displayName: 'Enable Learning',
						name: 'enableLearning',
						type: 'boolean',
						default: true,
						description: 'Whether to enable learning from this interaction',
					},
					{
						displayName: 'Max Tokens',
						name: 'maxTokens',
						type: 'number',
						default: 1000,
						description: 'Maximum tokens for the response',
					},
					{
						displayName: 'Temperature',
						name: 'temperature',
						type: 'number',
						default: 0.7,
						typeOptions: {
							minValue: 0,
							maxValue: 2,
							numberStepSize: 0.1,
						},
						description: 'Temperature for response generation (0-2)',
					},
					{
						displayName: 'Include Sources',
						name: 'includeSources',
						type: 'boolean',
						default: false,
						description: 'Whether to include source information in the response',
					},
				],
			},
			// Execute Task Parameters
			{
				displayName: 'Task Description',
				name: 'taskDescription',
				type: 'string',
				required: true,
				displayOptions: {
					show: {
						operation: ['executeTask'],
					},
				},
				default: '',
				placeholder: 'Describe the task to execute',
				description: 'Detailed description of the task to execute',
			},
			{
				displayName: 'Task Parameters',
				name: 'taskParameters',
				type: 'json',
				displayOptions: {
					show: {
						operation: ['executeTask'],
					},
				},
				default: '{}',
				description: 'Parameters for the task as JSON object',
			},
			{
				displayName: 'Task Priority',
				name: 'taskPriority',
				type: 'options',
				displayOptions: {
					show: {
						operation: ['executeTask'],
					},
				},
				options: [
					{
						name: 'Low',
						value: 'low',
					},
					{
						name: 'Normal',
						value: 'normal',
					},
					{
						name: 'High',
						value: 'high',
					},
					{
						name: 'Critical',
						value: 'critical',
					},
				],
				default: 'normal',
				description: 'Priority level for the task',
			},
			{
				displayName: 'Timeout (seconds)',
				name: 'timeout',
				type: 'number',
				displayOptions: {
					show: {
						operation: ['executeTask'],
					},
				},
				default: 300,
				typeOptions: {
					minValue: 1,
					maxValue: 3600,
				},
				description: 'Timeout for task execution in seconds',
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
				if (operation === 'processMessage') {
					const message = this.getNodeParameter('message', i) as string;
					const sessionId = this.getNodeParameter('sessionId', i) as string;
					const userId = this.getNodeParameter('userId', i) as string;
					const contextDataString = this.getNodeParameter('contextData', i) as string;
					const processingOptions = this.getNodeParameter('processingOptions', i) as any;

					let contextData = {};
					if (contextDataString) {
						try {
							contextData = JSON.parse(contextDataString);
						} catch (error) {
							throw new NodeOperationError(this.getNode(), 'Invalid JSON in context data');
						}
					}

					const body = {
						message,
						session_id: sessionId || undefined,
						user_id: userId || undefined,
						context: contextData,
						options: {
							enable_memory: processingOptions.enableMemory !== false,
							enable_learning: processingOptions.enableLearning !== false,
							max_tokens: processingOptions.maxTokens || 1000,
							temperature: processingOptions.temperature || 0.7,
							include_sources: processingOptions.includeSources || false,
						},
					};

					responseData = await this.helpers.requestWithAuthentication.call(
						this,
						'aiAssistantApi',
						{
							method: 'POST',
							url: '/process',
							body,
							json: true,
						},
					);
				} else if (operation === 'getHealth') {
					responseData = await this.helpers.requestWithAuthentication.call(
						this,
						'aiAssistantApi',
						{
							method: 'GET',
							url: '/health',
							json: true,
						},
					);
				} else if (operation === 'getSystemStatus') {
					responseData = await this.helpers.requestWithAuthentication.call(
						this,
						'aiAssistantApi',
						{
							method: 'GET',
							url: '/system/status',
							json: true,
						},
					);
				} else if (operation === 'executeTask') {
					const taskDescription = this.getNodeParameter('taskDescription', i) as string;
					const taskParametersString = this.getNodeParameter('taskParameters', i) as string;
					const sessionId = this.getNodeParameter('sessionId', i) as string;
					const userId = this.getNodeParameter('userId', i) as string;
					const contextDataString = this.getNodeParameter('contextData', i) as string;
					const taskPriority = this.getNodeParameter('taskPriority', i) as string;
					const timeout = this.getNodeParameter('timeout', i) as number;

					let taskParameters = {};
					if (taskParametersString) {
						try {
							taskParameters = JSON.parse(taskParametersString);
						} catch (error) {
							throw new NodeOperationError(this.getNode(), 'Invalid JSON in task parameters');
						}
					}

					let contextData = {};
					if (contextDataString) {
						try {
							contextData = JSON.parse(contextDataString);
						} catch (error) {
							throw new NodeOperationError(this.getNode(), 'Invalid JSON in context data');
						}
					}

					const body = {
						description: taskDescription,
						parameters: taskParameters,
						session_id: sessionId || undefined,
						user_id: userId || undefined,
						context: contextData,
						priority: taskPriority,
						timeout,
					};

					responseData = await this.helpers.requestWithAuthentication.call(
						this,
						'aiAssistantApi',
						{
							method: 'POST',
							url: '/tasks/execute',
							body,
							json: true,
							timeout: (timeout + 30) * 1000, // Add buffer to request timeout
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